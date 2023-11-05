import numpy as np
from scipy.stats import entropy
from scipy import ndimage
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
l1_dist = lambda x1,x2: np.sum(np.abs(x1 - x2), axis=tuple(range(len(x1.shape))[1:]))
l2_dist = lambda x1,x2: np.sum((x1-x2)**2, axis=tuple(range(len(x1.shape))[1:]))**.5

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def kl(x1, x2):
    assert x1.shape == x2.shape
    x1_2d_t = x1.transpose()
    x2_2d_t = x2.transpose()

    e = entropy(x1_2d_t, x2_2d_t)
    e[np.where(e==np.inf)] = 2
    return e

def reduce_precision_py(x, npp):
    """
    Reduce the precision of image, the numpy version.
    :param x: a float tensor, which has been scaled to [0, 1].
    :param npp: number of possible values per pixel. E.g. it's 256 for 8-bit gray-scale image, and 2 for binarized image.
    :return: a tensor representing image(s) with lower precision.
    """
    # Note: 0 is a possible value too.
    npp_int = npp - 1
    x_int = np.rint(x * npp_int)
    x_float = x_int / npp_int
    return x_float

def median_filter_py(x, width, height=-1):
    """
    Median smoothing by Scipy.
    :param x: a tensor of image(s)
    :param width: the width of the sliding window (number of pixels)
    :param height: the height of the window. The same as width by default.
    :return: a modified tensor with the same shape as x.
    """
    if height == -1:
        height = width
    var = ndimage.filters.median_filter(x, size=(1,width,height,1), mode='reflect')
    return torch.from_numpy(var)


def opencv_wrapper(imgs, opencv_func, argv):
    ret_imgs = []
    imgs_copy = imgs

    if imgs.shape[3] == 1:
        imgs_copy = np.squeeze(imgs)

    for img in imgs_copy:
        img_uint8 = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)
        ret_img = opencv_func(*[img_uint8]+argv)
        if type(ret_img) == tuple:
            ret_img = ret_img[1]
        ret_img = ret_img.astype(np.float32) / 255.
        ret_imgs.append(ret_img)
    ret_imgs = np.stack(ret_imgs)

    if imgs.shape[3] == 1:
        ret_imgs = np.expand_dims(ret_imgs, axis=3)

    return 

def bit_depth_py(x, bits):
    precisions = 2**bits
    return reduce_precision_py(x, precisions)

def non_local_means_color_py(imgs, search_window, block_size, photo_render):
    import cv2
    ret_imgs = opencv_wrapper(imgs, cv2.fastNlMeansDenoisingColored, [None,photo_render,photo_render,block_size,search_window])
    return ret_imgs


m = nn.Softmax(dim=1)

def get_distance(model, X1):
    X1_pred = m(model(X1))
    vals_squeezed = []

    X1_seqeezed_bit = bit_depth_py(X1.cpu(), 5)
    vals_squeezed.append(m(model((X1_seqeezed_bit).to(device))))
    X1_seqeezed_filter_median = median_filter_py(X1.cpu(), 2)
    vals_squeezed.append(m(model((X1_seqeezed_filter_median).to(device))))

    dist_array = []
    for val_squeezed in vals_squeezed:
        dist = np.sum(np.abs(X1_pred.cpu().detach().numpy() - val_squeezed.cpu().detach().numpy()), axis=tuple(range(len(X1_pred.shape))[1:]))
        dist_array.append(dist)

    dist_array = np.array(dist_array)
    return np.max(dist_array, axis=0)

def train_fs(model, X1, train_fpr):
    distances = get_distance(model,  X1)
    selected_distance_idx = int(np.ceil(len(X1) * (1-train_fpr)))
    threshold = sorted(distances)[selected_distance_idx-1]
    threshold = threshold
    return threshold

def get_distance_test(model,  X1):
    X1_pred = m(model(X1))
    vals_squeezed = []
    
    X1_seqeezed_bit = bit_depth_py(X1.detach().cpu(), 5)
    vals_squeezed.append(m(model((X1_seqeezed_bit).to(device))))
    X1_seqeezed_filter_median = median_filter_py(X1.detach().cpu(), 2)
    vals_squeezed.append(m(model((X1_seqeezed_filter_median).to(device))))

    dist_array = []
    for val_squeezed in vals_squeezed:
        dist = np.sum(np.abs(X1_pred.cpu().detach().numpy() - val_squeezed.cpu().detach().numpy()), axis=tuple(range(len(X1_pred.shape))[1:]))
        dist_array.append(dist)

    dist_array = np.array(dist_array)
    return np.max(dist_array, axis=0)

def test(model, X, threshold):
    distances = get_distance_test(model, X)
    Y_pred = distances > threshold
    return Y_pred, distances

def compute_distance(model, X):
    distances = get_distance_test(model, X)
    return distances

def get_tpr_fpr(true_labels, pred_labels):
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))

    AP = np.sum(true_labels)
    AN = np.sum(1-true_labels)

    tpr = TP/AP if AP>0 else np.nan
    fpr = FP/AN if AN>0 else np.nan

    return tpr, fpr, TP, AP, FP, AN

def evaluate_test(model, x_test, threshold):
    Y_all = np.concatenate([np.ones(len(x_test), dtype=bool)])
    Y_all_pred, Y_all_pred_score = test(model,  x_test, threshold)
    tpr, fpr, tp, ap, fp, an = get_tpr_fpr(Y_all, Y_all_pred)
    return tpr, fpr, tp, ap, fp, an