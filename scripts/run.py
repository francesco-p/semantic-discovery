"""
Charset: utf-8
indentation: 4 spaces
author : lakj
"""
import sys  
sys.path.insert(0, '../src')

import detector as de
import extractor as ex
import accumulator as ac
import utils
import metrics
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.segmentation import slic
import ipdb 
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('run.py')

############### PARAMS #############
N_KEYPOINTS = 4000                  
K = 7                               
RADIUS = 3                          
SIGMA = 3                           
W = 5                               
TAU = 1                             
N_SUPERPIXELS = 80                  
ALPHA = 1                           
INPUT_FILE = "../datasets/img01.png" 
OUTPUT_FILE = "../output/img01.png"  
####################################

logger.info("read image")
img = cv2.imread(INPUT_FILE, cv2.IMREAD_UNCHANGED)

logger.info("keypoints")
kpdetector = de.KeypointsDetector()
keypoints = kpdetector.canny(img, N_KEYPOINTS)

logger.info("descriptors")
extractor = ex.DescriptorExtractor()
descriptors = extractor.daisy(keypoints, img)

logger.info("splashes")
accumulator = ac.Accumulator(img)
# https://github.com/mariusmuja/flann/issues/143
FLANN_INDEX_KDTREE = 0
flann = cv2.FlannBasedMatcher(dict(algorithm = FLANN_INDEX_KDTREE, trees = 5), dict(checks=50))
matches = flann.knnMatch(descriptors, descriptors, k=K)

for i, m_list in enumerate(matches):

    o = (int(keypoints[i].pt[1]), int(keypoints[i].pt[0]))

    points = []
    rank = 1
    for m in m_list:

        d = (int(keypoints[m.trainIdx].pt[1]), int(keypoints[m.trainIdx].pt[0]))

        if utils.eu_dist(o, d) > RADIUS:
            points.append(d)
            accumulator.add_vote(o, d, rank, ksize=W)
            rank += 1

    accumulator.add_splash(o, points)


logger.info("threshold")
x, y = np.where(accumulator.accumulator > TAU)

vote_list = accumulator.votes.copy()
idx1 = np.nonzero(np.isin(vote_list[:, 0], x))[0]
idx2 = np.nonzero(np.isin(vote_list[idx1, 1], y))[0]
idx3 = idx1[idx2]
mask = np.zeros(vote_list.shape[0])

coords = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)

for coord in coords:
    idx4 = np.where(np.all(vote_list[idx3, 0:2]==coord, axis=1))[0]
    mask[idx3[idx4]] = 1

vote_list = vote_list[np.nonzero(mask)[0]]

logger.info("superpixels")
compactness = 25
superpixels = slic(img, n_segments=N_SUPERPIXELS, compactness=compactness, sigma=SIGMA)

logger.info("graph")
connections = []
for vote in vote_list:
    pi1 = vote[2:4]
    pi2 = vote[4:6]
    connections.append((superpixels[pi1[0], pi1[1]], superpixels[pi2[0], pi2[1]]))

n_clusters = superpixels.max()+1

adj_mat = np.zeros((n_clusters, n_clusters))
for c in connections:
    adj_mat[c[0], c[1]] += 1
    adj_mat[c[1], c[0]] += 1

np.fill_diagonal(adj_mat, 0)

logger.info("clustering")
best_partition_nodes, best_partition = utils.findBestPartition_alt(adj_mat, alpha=ALPHA, viz=False)
if best_partition_nodes == []:
    logger.error("change ALPHA")


logger.info("plot")
pred, rmask = utils.plot_spixel_segmentation(img, best_partition_nodes, superpixels) 

rmask = (rmask*255.).astype('uint8')
mask_over_img = cv2.addWeighted(rmask, 0.5, img, 0.5, 0)
cv2.imwrite(f"{OUTPUT_FILE}", mask_over_img)
        
