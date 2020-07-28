import cv2
import numpy as np

class KeypointsDetector:
    """ This class purpose is to provide a set of meaningful keypoints where to compute a the descriptors """

    def __init__(self):
        pass

    def canny(self, img, keypoints=8000):
        """ Generates source and destination keypoints by using Canny algorithm

        Parameters
        ----------
        img : ndarray
            the image itself where to compute the keypoints

        Returns
        -------
        list
            First element are the keypoints
        """

        height = img.shape[0]
        width = img.shape[1]
        src_keypoints = []

        # Conversion to gray image since goodFeaturesToTrack needs gray images
        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Source keypoints through Canny
        points = cv2.goodFeaturesToTrack(img_g, keypoints, 0.001, 3)
        for marker in points:
            src_keypoints.append(cv2.KeyPoint(x=marker.reshape(-1)[0], y=marker.reshape(-1)[1], _size=1))

        # Order from left to right, from top to bottom, this is to improve debugging
        src_keypoints = sorted(src_keypoints, key=lambda k: k.pt[0])
        src_keypoints = sorted(src_keypoints, key=lambda k: k.pt[1])

        return src_keypoints
