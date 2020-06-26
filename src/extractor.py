"""
This is temporary code, it is now under refactoring by the author. It is meant to be a working example for the reviewers
"""
import cv2
import numpy as np

class Extractor:
    """ This class purpose is to extract meaningful descriptors from an image given a set of hopefully useful keypoints """

    def __init__(self):
        pass

    def extract_daisy(self, src_keypoints, dst_keypoints, img):
        """ Given a set of src_points and dst_points it computes DAISY descriptors on those points

        Parameters
        ----------
        src_keypoints : list
            list of KeyPoint
        dst_keypoints : list
            list of KeyPoint
        img : ndarray
            the image itself where to compute the descriptors

        Returns
        -------
        tuple
            first two elements are the src_keypoints and the relative descriptors, the third and the fourth are dst_keypoints and the relative descriptors 
        """
        detector = cv2.xfeatures2d.DAISY_create(10,3,4,4)
        kp1, des1 = detector.compute(img, src_keypoints)
        kp2, des2 = detector.compute(img, dst_keypoints)
        return kp1, des1, kp2, des2
