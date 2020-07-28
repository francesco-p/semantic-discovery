import numpy as np
import time
from sklearn.metrics import jaccard_score
from scipy import stats
from collections import defaultdict


def get_metrics(pred, FILENAME, DATA_PATH):
    pred = pred.astype(np.int)
    
    binary_pred = (pred>0).astype(np.int)

    seg_cls = DATA_PATH + "/annotation/SegmentationClass/"
    gt_file = seg_cls+f'{FILENAME[-11:][:-4]}.npy'
    gt = np.load(gt_file)
    binary_gt = (gt>0).astype(np.int)

    obj_cls = DATA_PATH + "/annotation/SegmentationObject/"
    gt_file = obj_cls+f'{FILENAME[-11:][:-4]}.npy'
    gt_obj = np.load(gt_file)

    # Jaccard (intersection over union)
    jaccard = jaccard_score((gt>0).astype(np.int).flatten(), (pred>0).astype(np.int).flatten()) # INTERSECTIONOVERUNION

    # Total pattern (intersection)
    total_pattern = ((binary_pred + binary_gt )>1).sum()/(binary_gt ).sum() # INTERSECTION

    # Semantical consistency (precision)
    cs = np.unique(gt)
    ps = np.unique(pred)
    oss = np.unique(gt_obj)

    n_cs, n_ps, n_oss = cs.size, ps.size, oss.size

    p2o = np.zeros((n_ps, n_oss)).astype(np.int)
    c2o = np.zeros((n_cs, n_oss)).astype(np.int)

    for p, o, c in zip((binary_gt * pred).flatten(), (gt_obj * binary_pred).flatten(), (gt * binary_pred).flatten()):
        x = np.where(ps == p)[0][0]
        k = np.where(oss == o)[0][0]
        y = np.where(cs == c)[0][0]

        p2o[x, k] += 1
        c2o[y, k] += 1

    o2c = np.argmax(c2o.T, axis=1)

    cindices = np.argwhere(p2o>1)
    selfsegmentation = defaultdict(list)
    for ids in cindices:
        pattern, object_id = ids[0], ids[1]
        selfsegmentation[pattern].append(o2c[object_id])


    consistencies = []
    for key in selfsegmentation.keys():
        mode, count = stats.mode(selfsegmentation[key])
        consistencies.append((count / len(selfsegmentation[key]))[0])

    precision = np.array(consistencies).mean() # PRECISION

    # detected products (recall)
    tot_recall = 1 - (((p2o.sum(axis=0) == 0).sum())/(p2o.shape[1]-1)) # TOT_RECALL

    
    # (1) Quanti oggetti ci sono di classe semantica x?
    obj2cls = np.zeros((n_oss,n_cs)).astype(np.int)

    for o, c  in zip(gt_obj.flatten(), gt.flatten()):
        x = np.where(oss == o)[0][0]
        y = np.where(cs == c)[0][0]
        obj2cls[x,y] += 1

    obj_classes = np.where(obj2cls >0)[1] # (1)

    # (2) Quanto preciso è io pattern p con rispetto la classe c?
    mean_recalls = []
    for classe in np.unique(gt):
        if classe == 0:
            continue
        obj_ids = np.unique(gt_obj *(gt == classe))
        cls_cardinality = obj_ids.size-1
        #print(f"classe {classe}\n") 
        maxrecall = -1
        for pattern in np.unique(pred):
            if pattern == 0:
                continue
            pred_obj_ids = np.unique(gt_obj *(pred == pattern))

            n_cls = np.where(cs[obj_classes[pred_obj_ids]] == classe)[0].size
            recall = n_cls/cls_cardinality
            #print(f"[ P {pattern} ] recall  {n_cls}/{cls_cardinality} =  {recall}")
            if maxrecall < recall:
                maxrecall = recall
        # Mean for each class
        mean_recalls.append(maxrecall) # (2) 

    recall = np.array(mean_recalls).mean() # RECALL

    return jaccard,total_pattern,precision,tot_recall, recall
    
