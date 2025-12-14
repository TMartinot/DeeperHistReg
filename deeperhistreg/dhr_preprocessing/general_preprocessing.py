### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib
from typing import Union, Iterable, Sequence, Tuple
current_file = sys.modules[__name__]

### External Imports ###
import numpy as np
import torch as tc
import cv2


### Internal Imports ###
from dhr_utils import utils as u

########################

def basic_preprocessing(
    source : Union[tc.Tensor, np.ndarray],
    target : Union[tc.Tensor, np.ndarray],
    source_landmarks : Union[tc.Tensor, np.ndarray],
    target_landmarks : Union[tc.Tensor, np.ndarray],
    params : dict) -> Tuple[Union[tc.Tensor, np.ndarray], Union[tc.Tensor, np.ndarray], Union[tc.Tensor, np.ndarray], Union[tc.Tensor, np.ndarray], dict]:
    """
    TODO - documentation
    """
    postprocessing_params = dict()
    postprocessing_params['original_size'] = source.shape[0:2] if isinstance(source, np.ndarray) else (source.size(2), source.size(3))
    print(f"TM:gen_pre:32: pre: {source.shape=} | {target.shape=}")

    initial_resampling = params['initial_resampling']
    if initial_resampling:
        initial_resolution = params['initial_resolution']
        source_y_size, source_x_size, target_y_size, target_x_size = u.get_combined_size(source, target)
        initial_resample_ratio = u.calculate_resampling_ratio((source_x_size, target_x_size), (source_y_size, target_y_size), initial_resolution)
        initial_smoothing = max(initial_resample_ratio - 1, 0.1)
        new_size = (int(round(initial_resample_ratio * target_x_size)), int(round(initial_resample_ratio * target_y_size)))
        print(f"TM:gen_pre:resampling params: {initial_resample_ratio=} | {initial_smoothing=}")
        ugss = u.gaussian_smoothing(source, initial_smoothing)
        source = u.resample_tensor_to_size(ugss, new_size)
        ugst = u.gaussian_smoothing(target, initial_smoothing)
        target = u.resample_tensor_to_size(ugst, new_size)
        # FIXME TM >>> target = u.resample_tensor_to_size(ugst, new_size=)
        print(f"TM:Origin of bug (gaussian/resampling): gaussian {ugss.shape=} | {ugst.shape=} | resampling: {source.shape=} | {target.shape=}")
        postprocessing_params['initial_resample_ratio'] = initial_resample_ratio
        if source_landmarks is not None:
            source_landmarks = source_landmarks / initial_resample_ratio
        if target_landmarks is not None:
            target_landmarks = target_landmarks / initial_resample_ratio
    postprocessing_params['initial_resampling'] = initial_resampling
    print(f"TM:gen_pre:48: after init resampling: {source.shape=} | {target.shape=}")

    normalization = params['normalization']
    if normalization:
        source, target = u.normalize(source), u.normalize(target)
    print(f"TM:gen_pre:53: after norm.: {source.shape=} | {target.shape=}")

    convert_to_gray = params['convert_to_gray']
    if convert_to_gray:
        if params['flip_intensity']:
            source = 1 - u.convert_to_gray(source)
            target = 1 - u.convert_to_gray(target)
        else:
            source = u.convert_to_gray(source)
            target = u.convert_to_gray(target)
        print(f"TM:gen_pre:63: after gray conv: {source.shape=} | {target.shape=}")

        clahe = params['clahe']
        if clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            src = clahe.apply((source[0, 0].detach().cpu().numpy()*255).astype(np.uint8))
            trg = clahe.apply((target[0, 0].detach().cpu().numpy()*255).astype(np.uint8))
            source = tc.from_numpy((src.astype(np.float32) / 255)).to(source.device).unsqueeze(0).unsqueeze(0)
            target = tc.from_numpy((trg.astype(np.float32) / 255)).to(target.device).unsqueeze(0).unsqueeze(0)
        print(f"TM:gen_pre:72: after CLAHE: {source.shape=} | {target.shape=}")
    print(f"TM:gen_pre:73: after all: {source.shape=} | {target.shape=}")

    return source, target, source_landmarks, target_landmarks, postprocessing_params