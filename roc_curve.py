import matplotlib.pyplot as plt
import numpy as np
from main import compute_tpr_fpr

def max_roc_curve(benign, adv, step):
    fpr = []
    tpr = []
    minthres = float(min(benign))
    maxthres = float(max(benign))
    increment = (maxthres-minthres)/step
    i = minthres
    j = i
    while j < maxthres:
        temptpr, tempfpr = compute_tpr_fpr(i, j, adv, benign)
        fpr.append(tempfpr)
        tpr.append(temptpr)
        j += increment
    return fpr, tpr
            
def min_roc_curve(benign, adv, step):
    fpr = []
    tpr = []
    minthres = float(min(benign))
    maxthres = float(max(benign))
    increment = (maxthres-minthres)/step
    i = minthres
    j = maxthres
    while i < maxthres:
        temptpr, tempfpr = compute_tpr_fpr(i, j, adv, benign)
        fpr.append(tempfpr)
        tpr.append(temptpr)
        i += increment
    return fpr, tpr

def auroc(fpr, tpr):
    zipped_pairs = zip(fpr, tpr)
    sort = [x for x in sorted(zipped_pairs)]
    fpr, tpr = sort
    auroc = 0
    for i in range(1, len(fpr)):
        auroc += (fpr[i] - fpr[i-1]) * tpr[i]
    return auroc