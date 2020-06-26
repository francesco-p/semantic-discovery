"""
This is temporary code, it is now under refactoring by the author. It is meant to be a working example for the reviewers
"""
import cv2
import numpy as np

class KPDetector:
    """ This class purpose is to provide a set of meaningful keypoints where to compute a the descriptors """

    def __init__(self):
        pass



    def kp_by_canny(self, img, nsrc=8000, ndst=8000):
        """ Generates source and destination keypoints by using Canny algorithm

        Parameters
        ----------
        img : ndarray
            the image itself where to compute the keypoints

        Returns
        -------
        tuple
            First element are the source keypoints and the second element are the destination keypoints
        """

        height = img.shape[0]
        width = img.shape[1]
        src_keypoints = []
        dst_keypoints = []

        # Conversion to gray image since goodFeaturesToTrack needs gray images
        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Source keypoints through Canny
        points = cv2.goodFeaturesToTrack(img_g, nsrc, 0.001, 3)
        for marker in points:
            src_keypoints.append(cv2.KeyPoint(x=marker.reshape(-1)[0], y=marker.reshape(-1)[1], _size=1))

        # Destination keypoints through Canny
        points = cv2.goodFeaturesToTrack(img_g, ndst, 0.001, 1)
        for marker in points:
            dst_keypoints.append(cv2.KeyPoint(x=marker.reshape(-1)[0], y=marker.reshape(-1)[1], _size=1))

        # Order from left to right, from top to bottom, this is to improve debugging
        src_keypoints = sorted(src_keypoints, key=lambda k: k.pt[0])
        src_keypoints = sorted(src_keypoints, key=lambda k: k.pt[1])

        return src_keypoints, dst_keypoints
