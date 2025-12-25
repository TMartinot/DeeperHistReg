### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union, Sequence, Iterable, Tuple
import logging

### External Imports ###
import numpy as np
import torch as tc
import pyvips
import openslide

### Internal Imports ###
from loader import WSILoader, LoadMode
from dhr_utils import utils as u


class OpenSlideNativeLoader(WSILoader):
    """
    OpenSlide-backed WSI loader that preserves pyvips.Image outputs
    for downstream compatibility.
    """
    def __init__(self, image_path, mode=LoadMode.NUMPY):
        self.image_path = image_path
        self.mode = mode

        # Cheap: metadata only
        self.slide = openslide.OpenSlide(self.image_path)

        self.num_levels = self.slide.level_count
        self.resolutions = self.get_resolutions()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pil_region_to_vips(self, pil_img) -> pyvips.Image:
        """
        Convert PIL Image -> pyvips.Image
        """
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        width, height = pil_img.size

        # Scoped copy: released after function returns
        buf = pil_img.tobytes()

        return pyvips.Image.new_from_memory(
            buf,
            width,
            height,
            3,
            format="uchar"
        )

    def _read_region_as_vips(
        self,
        level: int,
        offset: Tuple[int, int],
        shape: Tuple[int, int],
    ) -> pyvips.Image:
        pil_img = self.slide.read_region(
            location=(offset[1], offset[0]),
            level=level,
            size=(shape[1], shape[0]),
        )
        return self._pil_region_to_vips(pil_img)

    def _read_level_as_vips(self, level: int) -> pyvips.Image:
        """
        Read entire level on demand.
        WARNING: this can be very large.
        """
        width, height = self.slide.level_dimensions[level]

        return self._read_region_as_vips(
            level=level,
            offset=(0, 0),
            shape=(height, width),
        )

    # ------------------------------------------------------------------
    # Public API (unchanged)
    # ------------------------------------------------------------------

    def get_num_levels(self) -> int:
        return self.num_levels

    def get_resolutions(self) -> Iterable[int]:
        return [
            (height, width, 3)
            for (width, height) in self.slide.level_dimensions
        ]

    def load(self) -> pyvips.Image:
        return self._read_level_as_vips(0)

    def get_best_level(self, resample_ratio: float) -> Tuple[pyvips.Image, int]:
        if self.num_levels == 1:
            return self.load(), 0

        org_h, org_w = self.resolutions[0][:2]
        desired_h = org_h * resample_ratio
        desired_w = org_w * resample_ratio

        level = 0
        for i, (h, w, _) in enumerate(self.resolutions):
            if desired_h > h or desired_w > w:
                break
            level = i

        return self._read_level_as_vips(level), level

    def update_resample_ratio(self, resample_ratio: float, level_to_use: int) -> float:
        org_h = self.resolutions[0][0]
        lvl_h = self.resolutions[level_to_use][0]
        return (org_h * resample_ratio) / lvl_h

    def resample(self, resample_ratio: float) -> Union[np.ndarray, tc.Tensor, pyvips.Image]:
        image, level = self.get_best_level(resample_ratio)

        if level > 0:
            resample_ratio = self.update_resample_ratio(
                resample_ratio, level
            )

        sigma = u.calculate_smoothing_sigma(resample_ratio)
        resized = (
            image.gaussblur(sigma)
                 .resize(resample_ratio, kernel="linear", vscale=resample_ratio)
        )

        if self.mode == LoadMode.NUMPY:
            return resized.numpy()
        elif self.mode == LoadMode.PYTORCH:
            return u.image_to_tensor(resized.numpy())
        elif self.mode == LoadMode.PYVIPS:
            return resized
        else:
            raise ValueError("Unsupported mode.")

    def load_region(self, level: int, offset: Tuple[int, int], shape: Tuple[int, int]) -> Union[np.ndarray, tc.Tensor, pyvips.Image]:
        if level >= self.num_levels:
            level = self.num_levels - 1
            logging.warning(f"Using level {level}")

        region = self._read_region_as_vips(level, offset, shape)

        if self.mode == LoadMode.NUMPY:
            return region.numpy()
        elif self.mode == LoadMode.PYTORCH:
            return u.image_to_tensor(region.numpy())
        elif self.mode == LoadMode.PYVIPS:
            return region
        else:
            raise ValueError("Unsupported mode.")

    def load_regions(self, level: int, offsets: Sequence[Tuple[int, int]], shape: Tuple[int, int]) -> Union[np.ndarray, tc.Tensor]:
        if level >= self.num_levels:
            level = self.num_levels - 1
            logging.warning(f"Using level {level}")

        h, w = shape
        out = np.empty((len(offsets), h, w, 3), dtype=np.uint8)

        for i, offset in enumerate(offsets):
            region = self._read_region_as_vips(level, offset, shape)
            out[i] = region.numpy()

        if self.mode == LoadMode.NUMPY:
            return out
        elif self.mode == LoadMode.PYTORCH:
            return tc.from_numpy(out)
        else:
            raise ValueError("Unsupported mode.")

    def load_level(self, level: int) -> Union[np.ndarray, tc.Tensor, pyvips.Image]:
        image = self._read_level_as_vips(level)

        if self.mode == LoadMode.NUMPY:
            return image.numpy()
        elif self.mode == LoadMode.PYTORCH:
            return u.image_to_tensor(image.numpy())
        elif self.mode == LoadMode.PYVIPS:
            return image
        else:
            raise ValueError("Unsupported mode.")

