import os
import cv2
import numpy as np
from typing import List, Tuple

def write_video(file_prefix: str, ext: str, fps: float, frame_size: Tuple[int], frames: List[np.uint8]):
    """Write frames to a video
    
    Args:
        - file_prefix: The file prefix without the extension
        - ext: The file extension (helps determine which fourcc codec to use)
        - fps: Video frame rate
        - frame_size: Tuple of (width, height)
        - frames: List of numpy frames, in order, to write to a video file
    """
    if ext == ".mp4": # Compressed
        # H.264 codec, works with .mp4, seems to change pixel colors, but plays in vscode
        fourcc = cv2.VideoWriter_fourcc('a','v','c','1')
    elif ext == ".avi": # Uncompressed
        # Works with .avi, does not change pixel colors
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    else:
        raise ValueError(f"Unrecognized extension {ext}")

    filename = file_prefix + ext
    video = cv2.VideoWriter(filename, fourcc, fps, frame_size)
    for frame in frames:
        video.write(frame)
    video.release()