"""
Charset: utf-8
indentation: spaces
author : lakj
"""

import sys
sys.path.append("../src")
import matplotlib
matplotlib.use('Agg')
import detector as de
import extractor as ex
import accumulator as ac
import utils
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import networkx as nx
from skimage.segmentation import slic
import time
import pandas as pd
import fcntl
import traceback





# Algorithm Parameters

NSRC_POINTS = 9000 # must be NSRC == NDST, this is here only for old tests...
NDST_POINTS = 9000
K = 15
RADIUS = 30
THRESHOLD = 5
N_SEGMENTS = 150
MIN_CL = 2
MIN_CL2 = 2


#######################  FOLDER/FILE NAMES  #####################
DATA_PATH = "../datasets/Faces_collage/"
RES_PATH = "./results/Faces_collage/"
GT_FOLDER = "../datasets/Faces_collage_gt/"
if not os.path.exists(RES_PATH):
    os.makedirs(RES_PATH)

#######################  CSV COLUMNS  ###########################
CSV_FILE = "faces_collage.csv"
COLUMNS_ROW = "FILENAME,NSRC_POINTS,NDST_POINTS,K,RADIUS,THRESHOLD,N_SEGMENTS,MIN_CL,MIN_CL2,TIME,PRECISION,FALLBACK_PRECISION\n"
if not os.path.exists(RES_PATH + CSV_FILE):
    with open(RES_PATH + CSV_FILE, "w") as g:
        g.write(COLUMNS_ROW)

filenames = sorted(os.listdir(DATA_PATH))

for FILE_ID, FILENAME in enumerate(filenames):

    try:

        start = time.time()

        ########################  Extraction  ###########################
        print(f"{FILENAME} ", end="", flush=True)

        # Reads img
        img = cv2.imread(DATA_PATH+FILENAME, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Detects hotspots
        kpdetector = de.KPDetector()
        src_keypoints, dst_keypoints = kpdetector.kp_by_canny(img, NSRC_POINTS,NDST_POINTS)

        # Extracts the descriptors from previous detected points
        extractor = ex.Extractor()
        kp1, des1, kp2, des2 = extractor.extract_daisy(src_keypoints, dst_keypoints, img)

        ########################  Accumulator  ##########################
        print("flann ", end="", flush=True)

        # Declaration of the Accumulator
        accum_cls = ac.Accumulator(img)

        
        FLANN_INDEX_KDTREE = 0
        flann = cv2.FlannBasedMatcher(dict(algorithm = FLANN_INDEX_KDTREE, trees = 5), dict(checks=50))
        matches = flann.knnMatch(des1,des2,k=K)

        # Match comparisons
        for i, m_list in enumerate(matches):

            o = (int(kp1[i].pt[1]), int(kp1[i].pt[0]))

            points = []
            rank = 1
            for m in m_list:

                d = (int(kp2[m.trainIdx].pt[1]), int(kp2[m.trainIdx].pt[0]))

                # Removes the points near the source
                if utils.eu_dist(o, d) > RADIUS:

                    points.append(d)
                    accum_cls.add_vote(o, d, rank**2, ksize=11)
                    rank += 1

            accum_cls.add_splash(o, points)
                    
        #############  Accumulator thresholding  ########################
        print("thresholding ", end="", flush=True)

        x, y = np.where(accum_cls.accumulator > THRESHOLD)

        vote_list = accum_cls.votes.copy()
        idx1 = np.nonzero(np.isin(vote_list[:,0], x))[0]
        idx2 = np.nonzero(np.isin(vote_list[idx1,1], y))[0]
        idx3 = idx1[idx2]
        mask = np.zeros(vote_list.shape[0])

        coords = np.concatenate((x.reshape(-1,1), y.reshape(-1,1)), axis=1)

        for coord in coords:
            idx4 = np.where(np.all(vote_list[idx3,0:2] == coord, axis=1))[0]
            mask[idx3[idx4]] = 1

        vote_list = vote_list[np.nonzero(mask)[0]]

        ####################  Superpixel  ######################
        print("superpixel ", end="", flush=True)

        segments = slic(img, n_segments = N_SEGMENTS, compactness=25, sigma = 7)

        ####################  Graph creation  ####################
        print("graph ", end="", flush=True)

        connections = []
        for vote in vote_list:
            pi1 = vote[2:4]
            pi2 = vote[4:6]
            connections.append((segments[pi1[0],pi1[1]],segments[pi2[0],pi2[1]]))

        n_clusters = segments.max()+1

        adj_mat = np.zeros((n_clusters, n_clusters))
        for c in connections:
            adj_mat[c[0], c[1]] += 1
            adj_mat[c[1], c[0]] += 1

        np.fill_diagonal(adj_mat, 0)

        ################  Clustering algorithm over graph  #####################
        print("k-core ", end="", flush=True)

        flag = True
        best_partition_nodes, best_partition = utils.findBestPartition_alt(adj_mat,min_cl=MIN_CL, viz=False)

        if best_partition_nodes == []:

            flag = False

        final_partition_nodes = best_partition_nodes

        ###################  Measures  ####################
        if flag:
            
            end = time.time()
            TIME = end - start
            print(f"elapsed {TIME:.2f} sec")
            
            pred = utils.get_pred(img, final_partition_nodes, segments)#, path=RES_PATH+FILENAME)    

            gtid = FILENAME[-9:].split(".")[0]
            gt = np.load(GT_FOLDER + f"{gtid}.npy")

            fallback_max_precision = -1
            max_precision = -1
            for p in np.unique(pred):
                if p == 0:
                    continue
                tmp1 = (pred==p)
                tmp2 = gt>0
                intersection = (tmp1*tmp2)>0

                index = np.count_nonzero(intersection) / np.count_nonzero(tmp1)
                precision = (np.unique(intersection*gt).size -1) / (np.unique(gt).size -1) # precision == recall

                if (index >= 0.8) and (precision > max_precision):
                    max_precision = precision

                if precision > fallback_max_precision:
                    fallback_max_precision = precision

            new_entry = f"{FILENAME},{NSRC_POINTS},{NDST_POINTS},{K},{RADIUS},{THRESHOLD:.4f},{N_SEGMENTS},{MIN_CL},{MIN_CL2},{TIME:.4f},{max_precision:.4f},{fallback_max_precision:.4f}\n"

            with open(RES_PATH + CSV_FILE, "a") as g:
                fcntl.flock(g, fcntl.LOCK_EX)
                g.write(new_entry)
                fcntl.flock(g, fcntl.LOCK_UN)

            pred, rmask = utils.plot_spixel_segmentation(img, best_partition_nodes, segments) 
            plt.figure(figsize=(15,15))
            plt.imshow(img)
            plt.imshow(rmask, alpha=0.5)
            plt.axis('off')
            plt.savefig(RES_PATH + f"{FILENAME[:-4]}.png", dpi=300)
            plt.close()

        else:
            print(f"[ X ] {FILENAME} No partitions found")

    except Exception as e:
        print(f"[ FAIL ] {repr(e)} {traceback.print_exc()}")
