"""MediaPipe 1.0 FaceLandmarker setup that stays open on macOS 1.0.1."""

import sys

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def _version_tuple() -> tuple[int, ...]:
    parts: list[int] = []
    for piece in mp.__version__.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts[:3])


def uses_metal_delegate() -> bool:
    """1.0.1's macOS CPU graph aborts in DrishtiMetalHelper before inference.

    The face-detector calculator always opens a Metal helper, and 1.0.1 no
    longer installs the GPU service when the delegate is CPU. The GPU delegate
    installs that service. Metal then rejects SRGB frames, so images must be
    SRGBA.
    """
    return sys.platform == "darwin" and _version_tuple() >= (1, 0, 1)


def create_face_landmarker(model_asset_path) -> vision.FaceLandmarker:
    delegate = (
        python.BaseOptions.Delegate.GPU
        if uses_metal_delegate()
        else python.BaseOptions.Delegate.CPU
    )
    base_options = python.BaseOptions(
        model_asset_path=str(model_asset_path),
        delegate=delegate,
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def mp_image_from_frame(frame: np.ndarray) -> mp.Image:
    data = np.ascontiguousarray(frame)
    if data.dtype != np.uint8:
        data = np.ascontiguousarray(data.astype(np.uint8))
    if uses_metal_delegate() and data.ndim == 3 and data.shape[2] == 3:
        alpha = np.full((data.shape[0], data.shape[1], 1), 255, dtype=np.uint8)
        data = np.concatenate((data, alpha), axis=2)
        return mp.Image(image_format=mp.ImageFormat.SRGBA, data=data)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=data)
