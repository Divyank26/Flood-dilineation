import numpy as np
import cv2


def create_overlay(vv, mask):

    vv = (vv * 255).astype(np.uint8)

    vv = cv2.cvtColor(
        vv,
        cv2.COLOR_GRAY2BGR
    )

    flood = np.zeros_like(vv)

    flood[:, :, 2] = mask * 255

    overlay = cv2.addWeighted(
        vv,
        0.7,
        flood,
        0.3,
        0
    )

    return overlay