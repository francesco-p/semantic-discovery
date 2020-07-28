import numpy as np
import scipy.stats as st
import utils

class Accumulator:
    """ This class (is a mess) purpose is to hold the splash accumulator """

    def __init__(self, img):

        self.img_h = img.shape[0]
        self.img_w = img.shape[1]
        
        # A splash graph accumulator needs double of the image plane
        self.accumulator = np.zeros((self.img_h*2-1, self.img_w*2-1))
        self.accumulator_center = self.img_h-1, self.img_w-1
        
        self.votes= []
        self.splashes = {}

        self.init = True
        

    def add_vote(self, origin, destination, rank, ksize=31, ksig=3):
        
        # Generate the kernel to vote
        k = utils.gkern(ksize, ksig)
        k /= k.max()
        k *= 1/rank
        offs = k.shape[0]//2
        
        # Centering the origin
        dy = -1*origin[0]
        dx = -1*origin[1]

        accu_point = (destination[0]+dy+self.accumulator_center[0], destination[1]+dx+self.accumulator_center[1])

        # Adds the splash to the accumulator
        if accu_point != self.accumulator_center:
            tmp = self.accumulator[accu_point[0]-offs:accu_point[0]+offs+1, accu_point[1]-offs:accu_point[1]+offs+1]
            self.accumulator[accu_point[0]-offs:accu_point[0]+offs+1, accu_point[1]-offs:accu_point[1]+offs+1] += k[:tmp.shape[0], :tmp.shape[1]]

            # V set
            # currently it's saving only the central vote for better memory vs precision...
            if self.init:
                self.votes = np.array([[accu_point[0], accu_point[1], origin[0], origin[1],destination[0], destination[1]]])
                self.init = False
            else:
                self.votes = np.concatenate((self.votes, np.array([[accu_point[0], accu_point[1], origin[0], origin[1], destination[0], destination[1]]])), axis=0)

    
    def add_splash(self, origin, endpoints):
        
        # Centering the origin
        dy = -1*origin[0]
        dx = -1*origin[1]

        # Centering all the other endpoints
        npoints = []
        for p in endpoints:
            accu_point = (p[0]+dy+self.accumulator_center[0], p[1]+dx+self.accumulator_center[1])
            npoints.append(accu_point)
        
        # Save the splash cohordinate is the key, at position 0 you have the endpoints in img reference
        # at position 1 you have the src endpoints in accumulator reference 
        self.splashes[origin] = endpoints, npoints


