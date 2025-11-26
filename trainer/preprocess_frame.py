"""
Frame preprocessing utilities.
Note: This module is kept for compatibility but is not used with vector observations.
For image-based environments, use the resize_frame function.

Requires opencv-python: pip install opencv-python
"""

try:
    import cv2
except ImportError:
    cv2 = None  # Will raise error if used without opencv-python

import numpy as np


def resize_frame(frame):
    """
    Preprocess frame for CNN input (for image-based environments).
    This function is not used with the vector observation Pong environment.

    Args:
        frame: Input frame image

    Returns:
        Preprocessed frame (84x84 grayscale)

    Raises:
        ImportError: If opencv-python is not installed
    """
    if cv2 is None:
        raise ImportError(
            "opencv-python is required for frame preprocessing. "
            "Install it with: pip install opencv-python"
        )

    # Crop and convert to grayscale
    frame = frame[30:-12, 5:-4]
    frame = np.average(frame, axis=2)
    # Resize to 84x84
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_NEAREST)
    frame = np.array(frame, dtype=np.uint8)
    return frame
