"""Video and calibration I/O."""

from os import path as opath

import cv2
import numpy as np


def load_calibration(root: str) -> tuple[np.ndarray, np.ndarray]:
    """Read K (3x3) and D (k1, k2, p1, p2, k3) from root/K.csv and root/D.csv."""
    K = np.loadtxt(opath.join(root, 'K.csv'), delimiter=',')
    D = np.loadtxt(opath.join(root, 'D.csv'), delimiter=',').ravel()
    return K, D


def load_frames(video_path: str) -> list[np.ndarray]:
    """Read every frame of a video as BGR uint8."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def undistort_frames(frames: list[np.ndarray], K: np.ndarray, D: np.ndarray) -> list[np.ndarray]:
    """Undistort keeping the same K, so K stays valid for the undistorted frames."""
    return [cv2.undistort(f, K, D) for f in frames]


def undistort_valid_mask(K: np.ndarray, D: np.ndarray, shape: tuple, erode_px: int = 6) -> np.ndarray:
    """Pixels of an undistorted frame that came from inside the original image.

    Undistortion leaves black wedges at the borders; their corners sit at the same
    pixel in every frame, so detectors fire on them and produce fake static matches.
    """
    white = np.full(shape[:2], 255, np.uint8)
    valid = np.uint8(cv2.undistort(white, K, D) == 255) * 255
    return cv2.erode(valid, np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8))
