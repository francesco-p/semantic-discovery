"""
Charset: utf-8
indentation: spaces
author : lakj
"""
import sys  
sys.path.insert(0, '../src')
import matplotlib
matplotlib.use('Agg')

import detector as de
import extractor as ex
import accumulator as ac
import utils
import metrics
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import networkx as nx
from skimage.segmentation import slic
import time
import pandas as pd
import fcntl
from sklearn.model_selection import ParameterGrid
import argparse
import traceback




parser = argparse.ArgumentParser()

# Folders and files
parser.add_argument('--PROJECT_NAME', required=True, choices=['semantic_lvl1','semantic_lvl2'],help='Project image folder')

# Algorithm Parameters
parser.add_argument('--NSRC_POINTS', nargs='+', type=int, default=[4000], help='NSRC_POINTS List') 
parser.add_argument('--NDST_POINTS', nargs='+', type=int, default=[4000], help='NDST_POINTS List')    
parser.add_argument('--N_SEGMENTS', nargs='+', type=int, default=[100,400,800,1200,1600,2000], help='Segments Superpixel')    
parser.add_argument('--COMPACTNESS', nargs='+', type=float, default=[10.0,25.0], help='Compactness superpixel')    
parser.add_argument('--SIGMA', nargs='+', type=int, default=[5,7], help='Sigma Superpixel')    
parser.add_argument('--K', nargs='+', type=int, default=[7], help='K List')    
parser.add_argument('--RADIUS', nargs='+', type=int, default=[70,120,200], help='RADIUS List')    
parser.add_argument('--THRESHOLD', nargs='+', type=float, default=[3.0], help='THRESHOLD List')    
parser.add_argument('--MIN_CL', nargs='+', type=int, default=[2], help='MIN_CL List')    

args = parser.parse_args()



#######################  FOLDER/FILE NAMES  #####################
DATA_PATH = f"../datasets/{args.PROJECT_NAME}/"
RES_PATH = f"./results/{args.PROJECT_NAME}/"
if not os.path.exists(RES_PATH):
    os.makedirs(RES_PATH)

CSV_FILE = f"{args.PROJECT_NAME}.csv"

#######################  CSV COLUMNS  ###########################
COLUMNS_ROW = "FILENAME,NSRC_POINTS,NDST_POINTS,K,RADIUS,THRESHOLD,N_SEGMENTS,MIN_CL,MIN_CL2,TIME,JACCARD,TOTAL_PATTERN,MU_CONSISTENCIES,TOT_RECALL,RECALL\n"
if not os.path.exists(RES_PATH + CSV_FILE):
    with open(RES_PATH + CSV_FILE, "w") as g:
        g.write(COLUMNS_ROW)


########################  PARAMETERS  ###########################
param_grid = {
              'NSRC_POINTS': args.NSRC_POINTS,
              'NDST_POINTS' : args.NDST_POINTS,
              'K' : args.K,
              'RADIUS' : args.RADIUS,
              'THRESHOLD' : args.THRESHOLD,
              'COMPACTNESS' : args.COMPACTNESS,
              'SIGMA' : args.SIGMA,
              'N_SEGMENTS' : args.N_SEGMENTS,
              'MIN_CL' : args.MIN_CL
              }
grid = ParameterGrid(param_grid)

lengrid = len(grid)

filenames = [f for f in os.listdir(DATA_PATH) if f[-3:]=='png']

totalruns = lengrid*len(filenames)

print(f"Total combination of parameters: {lengrid} filenames : {len(filenames)} tot:{totalruns}")
                
# MAIN LOOP
for RUN_ID,params in enumerate(grid):
    NSRC_POINTS = params['NSRC_POINTS']
    NDST_POINTS = params['NDST_POINTS']
    K =  params['K']
    N_SEGMENTS = params['N_SEGMENTS']
    RADIUS = params['RADIUS']
    THRESHOLD = params['THRESHOLD']
    COMPACTNESS = params['COMPACTNESS']
    SIGMA = params['SIGMA']
    MIN_CL = params['MIN_CL']
    MIN_CL2 = 2

    
    
    for FILE_ID, FILENAME in enumerate(filenames):
        
        start = time.time()

        try:
            ########################  Extraction  ###########################
            print(f"[ {RUN_ID}/{lengrid} ]({FILE_ID}/{len(filenames)}) Extraction ", end="", flush=True)

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

            # https://github.com/mariusmuja/flann/issues/143
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
            #############  Accumulator thresholding  ###############
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

            segments = slic(img, n_segments = N_SEGMENTS, compactness = COMPACTNESS, sigma = SIGMA)
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
            ################  Clustering algorithm  #####################
            print("k-core ", end="", flush=True)

            flag = True
            best_partition_nodes, best_partition = utils.findBestPartition_alt(adj_mat,min_cl=MIN_CL, viz=False)

            if best_partition_nodes == []:

                flag = False

            #else:

                #G = nx.from_numpy_array(best_partition)
                #final_partition_nodes = []
                #for i,c in enumerate(best_partition_nodes):
                    #sg = G.subgraph(c)
                    #edges,weights = zip(*nx.get_edge_attributes(sg,'weight').items())
                    #sg_density = len(edges) / len(sg.nodes)
                    #weights = np.array(weights)
                    #weights /= weights.max()
                    #weights *= 10
                    #sg_mat = np.array(nx.to_numpy_matrix(G, nodelist=sg.nodes))
                    #best_partition_nodes_C, best_partition_C = utils.findBestPartition(sg_mat, min_cl=MIN_CL2, viz=False)

                    #for cl in best_partition_nodes_C:
                        #final_partition_nodes.append(np.array(sg.nodes)[np.array(cl)])
                        
                #if len(final_partition_nodes) == 0:
                    #final_partition_nodes = best_partition_nodes
            final_partition_nodes = best_partition_nodes
            ###################  MEASURES  ####################
            print("plot ", end="", flush=True)

            if flag:
                
                end = time.time()
                TIME = end - start
                print(f"elapsed {TIME:.2f} sec")
                
                pred = utils.get_pred(img, final_partition_nodes, segments)#, path=RES_PATH+FILENAME)    
                
                jaccard,total_pattern,mu_consistencies,tot_recall,recall = metrics.get_metrics(pred, FILENAME, DATA_PATH)
                ####################  LOGGING  ####################
                new_entry = f"{FILENAME},{NSRC_POINTS},{NDST_POINTS},{K},{RADIUS},{THRESHOLD:.4f},{N_SEGMENTS},{MIN_CL},{MIN_CL2},{TIME:.4f},{jaccard:.4f},{total_pattern:.4f},{mu_consistencies:.4f},{tot_recall:.4f},{recall:.4f}\n"

                with open(RES_PATH + CSV_FILE, "a") as g:
                    fcntl.flock(g, fcntl.LOCK_EX)
                    g.write(new_entry)
                    fcntl.flock(g, fcntl.LOCK_UN)
                
                # Figure
                if recall > 0.8  and mu_consistencies > 0.80:
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
        
