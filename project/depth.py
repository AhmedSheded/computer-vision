import numpy as np


def createMedianMask(disparityMap, validDepthMask, rect=None):
    """returns a mask selactiong the median layer, plus shaddows"""
    if rect is not None:
        x, y, w, h = rect
        disparityMap = disparityMap[y: y+h, x:x+w]
        validDepthMask = validDepthMask[y:y+h, x:x+w]

    median = np.median(disparityMap)
    return np.where((validDepthMask == 0) | (abs(disparityMap - median) < 12), 255, 0).astype(np.uint8)