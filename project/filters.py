import cv2
import numpy as np
import utils


def strokeEdges(src, dst, blurKsize=7, edgKsize=5):
    if blurKsize >= 3:
        blurredSrc = cv2.meidanBlur(src, blurKsize)
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize=edgKsize)
    normalizedInversAlpha = (1.0 / 255) * (255 - graySrc)
    # print(normalizedInversAlpha)
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalizedInversAlpha
    cv2.merge(channels, dst)


class VConvolutionFilter(object):
    """A filter that applies a vonvlution to V (orall of BGR)."""
    def __init__(self, kernal):
        self._kernal = kernal

    def apply(self, src, dst):
        """Apply the filter with a BGR or gray source/destinatin."""
        cv2.fillter2D(src, -1, self._kernal, dst)


class SharpenFilter(VConvolutionFilter):
    """A sharpen filter with a 1-pixel radius."""
    def __init__(self):
        kernal = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernal)


class FindEdgesFilter(VConvolutionFilter):
    """An edge-finding filter with a 1-pixel radius."""
    def __init__(self):
        kernal = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernal)