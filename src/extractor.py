import cv2
import numpy as np

class DescriptorExtractor:
    """ This class purpose is to extract meaningful descriptors from an image given a set of hopefully useful keypoints """

    def __init__(self):
        pass

    def daisy(self, keypoints, img):
        """ Given a set of src_points it computes DAISY descriptors on those points

        Parameters
        ----------
        keypoints : list
            list of KeyPoint
        img : ndarray
            the image itself where to compute the descriptors

        Returns
        -------
        tuple
            the two elements are the src_keypoints and the relative descriptors
        """
        detector = cv2.xfeatures2d.DAISY_create(10,3,4,4)
        keypoints, descriptors = detector.compute(img, keypoints)
        return descriptors
