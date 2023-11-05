import os, cv2,itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image

# pytorch libraries
import torch
from torch import optim,nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import models,transforms

# sklearn libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import gc

# to make the results are reproducible
np.random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed(10)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import GradientShap
from captum.attr import GuidedGradCam
from captum.attr import LimeBase
from captum.attr import KernelShap
from captum.attr import Occlusion
from captum.attr import visualization as viz
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

def compute_mean_abs_dev(attr): 
    scores = []
    for i in range(len(attr)):
        a = attr[i].flatten()
        avg = np.mean(a)
        deviation = a - avg 
        absolute_deviation = np.abs(deviation)
        result = np.mean(absolute_deviation)
        scores.append(result)
    return scores    

def compute_median_abs_dev(attr): 
    scores = []
    for i in range(len(attr)):
        a = attr[i].flatten()
        med = np.median(a)
        deviation = a - med 
        abs_deviation = np.abs(deviation)
        result = np.median(abs_deviation)
        scores.append(result)
    return scores 

def compute_iqr(attr):
    #inter-quartile range
    scores = []
    for i in range(len(attr)):
        a = attr[i].flatten()
        score_75 = np.percentile(a, 75)
        score_25 = np.percentile(a, 25)
        score_qt = score_75 - score_25
        scores.append(score_qt)
    return scores
    
def compute_coef_var(attr):
    scores = []
    for i in range(len(attr)):
        a = attr[i].flatten()
        m = np.mean(a)
        st = np.std(attr[i])
        sc = m/st
        scores.append(sc)
    return scores

def compute_coef_iqr(attr):
    scores = []
    for i in range(len(attr)):
        a = attr[i].flatten()
        score_75 = np.percentile(a, 75)
        score_25 = np.percentile(a, 25)
        score_qt = (score_75 - score_25)/(score_75 + score_25)
        scores.append(score_qt)
    return scores
    
    
def get_attributions(algorithm, model, data_loader, num_batches):
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        model.cuda()
    dataiter = iter(data_loader)
    xai = algorithm
    attr = []
    for i in range(min(num_batches, len(data_loader))):
        images, labels = next(dataiter)
        labels = labels.to(device)
        for ind in range(len(images)):
            gc.collect()
            torch.cuda.empty_cache()
            input = images[ind].unsqueeze(0)
            input = input.to(device)
            # x_logits = model(input).max(1)
            a_batch = xai.attribute(input, target=labels[ind]).sum(axis=1).cpu().detach().numpy()
            attr.append(a_batch)
            gc.collect()
            torch.cuda.empty_cache()
        print("Attribution of batch " + str(i) + " complete")
    return attr
    

def get_stats(attr):
    medianAbs = []
    meanAbs = []
    iqr = []
    coef_var=[]
    coef_iqr = []
    
    for i in attr:
        meanAbs += compute_mean_abs_dev(i)
        medianAbs += compute_median_abs_dev(i)
        iqr += compute_iqr(i)
        coef_var += compute_coef_var(i)
        coef_iqr += compute_coef_iqr(i)
        
    return medianAbs, meanAbs, iqr, coef_var, coef_iqr  
                   
def benign_attr(algorithm, model, val_loader, num_batches):
    attr = []
    dataiter = iter(val_loader)
    xai = algorithm
    for i in range(min(num_batches, len(val_loader))):
        images, labels = next(dataiter)
        images = images.to(device)
        labels = labels.to(device)
        for ind in range(len(images)):
            gc.collect()
            torch.cuda.empty_cache()
            input = images[ind].unsqueeze(0)
            _, x_logits = model(input).max(1)
            a_batch_benign = xai.attribute(input, target=x_logits).sum(axis=1).cpu().detach().numpy()
            attr.append(a_batch_benign)
        print("Attribution for batch " + str(i) + " complete")
    return attr

def benign_attribution(algorithm, model, val_loader, num_batches):
    attr = benign_attr(algorithm, model, val_loader, num_batches)
    return get_stats(attr)

def run_benign_noise(algorithm, model, val_loader, num_batches, spread):
    attr = []
    dataiter = iter(val_loader)
    xai = algorithm
    for i in range(min(num_batches, len(val_loader))):
        images, labels = next(dataiter)
        images = images.to(device)
        labels = labels.to(device)
        for ind in range(len(images)):
            gc.collect()
            torch.cuda.empty_cache()
            original = images[ind].unsqueeze(0)
            _, y_original = model(original).max(1)
            attr_original = xai.attribute(original, target=y_original).sum(axis=1).cpu().detach().numpy()

            x = original.data.cpu().numpy()
            stdev = spread * (np.max(x)-np.min(x))
            noise = np.random.normal(0, stdev, x.shape).astype(np.float32)
            x_plus_noise = x + noise
            x_plus_noise = np.clip(x_plus_noise, 0, 1)
            noisy = torch.from_numpy(x_plus_noise).cpu()
            attr_noisy = xai.attribute(noisy, target=y_original).sum(axis=1).cpu().detach().numpy()

            a_batch_benign = np.linalg.norm(attr_noisy.flatten()-attr_original.flatten(),ord=1 )
            attr.append(a_batch_benign)
        print("Attribution for batch " + str(i) + " complete")
    return attr

def run_benign(algorithm, model, val_loader, num_batches):
    gc.collect()
    torch.cuda.empty_cache()
                   
    if torch.cuda.is_available():
        model.cuda()
    
    dataiter = iter(val_loader)
    xai = algorithm
    
    res = []
    
    medianAbs, meanAbs, iqr, coef_var, coef_iqr = benign_attribution(algorithm, model, val_loader, num_batches)
    
    res.append(medianAbs)
    res.append(meanAbs)
    res.append(iqr)
    res.append(coef_var)
    res.append(coef_iqr)
    
    return res
    
def run_pgd(algorithm, model, val_loader, eps, num_batches):
    gc.collect()
    torch.cuda.empty_cache()
                   
    if torch.cuda.is_available():
        model.cuda()
    
    dataiter = iter(val_loader)
    xai = algorithm
    
    res = []
    
    medianAbs_bena = []
    meanAbs_bena = []
    iqr_bena = []
    coef_var_bena =[]
    coef_iqr_bena = []
    
    for i in range(min(num_batches, len(val_loader))):
        images, labels = next(dataiter)
        images = images.to(device)
        labels = labels.to(device)
        for ind in range(len(images)):
            gc.collect()
            torch.cuda.empty_cache()
            input = images[ind].unsqueeze(0)
            alpha = eps/10
            steps = int(alpha*eps)
            images_pgd = projected_gradient_descent(model, input, eps, alpha, steps, np.inf)
            _, y_pred_pgd = model(images_pgd).max(1)
            a_batch_attack = xai.attribute(inputs=images_pgd, target=y_pred_pgd).sum(axis=1).cpu().detach().numpy()
            meanAbs_bena += compute_mean_abs_dev(a_batch_attack)
            medianAbs_bena += compute_median_abs_dev(a_batch_attack)
            iqr_bena += compute_iqr(a_batch_attack)
            coef_var_bena += compute_coef_var(a_batch_attack)
            coef_iqr_bena += compute_coef_iqr(a_batch_attack)
            gc.collect()
            torch.cuda.empty_cache()
        print("Attribution for batch " + str(i) + " complete")
    
    res.append(medianAbs_bena)
    res.append(meanAbs_bena)
    res.append(iqr_bena)
    res.append(coef_var_bena)
    res.append(coef_iqr_bena)
    print("PGD Calculation Complete")
    
    return res

def run_pgd_noise(algorithm, model, val_loader, eps, num_batches, spread):
    gc.collect()
    torch.cuda.empty_cache()
                   
    if torch.cuda.is_available():
        model.cuda()
    
    dataiter = iter(val_loader)
    xai = algorithm
    
    res = []
    
    for i in range(min(num_batches, len(val_loader))):
        images, labels = next(dataiter)
        images = images.to(device)
        labels = labels.to(device)
        for ind in range(len(images)):
            gc.collect()
            torch.cuda.empty_cache()
            input = images[ind].unsqueeze(0)
            alpha = eps/10
            steps = int(alpha*eps)
            images_pgd = projected_gradient_descent(model, input, eps, alpha, steps, np.inf)
            original = images_pgd
            _, y_original = model(original).max(1)
            attr_original = xai.attribute(original, target=y_original).sum(axis=1).cpu().detach().numpy()

            x = images_pgd.data.cpu().numpy()
            stdev = spread * (np.max(x)-np.min(x))
            noise = np.random.normal(0, stdev, x.shape).astype(np.float32)
            x_plus_noise = x + noise
            x_plus_noise = np.clip(x_plus_noise, 0, 1)
            noisy = torch.from_numpy(x_plus_noise).cpu()
            attr_noisy = xai.attribute(noisy, target=y_original).sum(axis=1).cpu().detach().numpy()
            
            a_batch_attack = np.linalg.norm(attr_noisy.flatten()-attr_original.flatten(),ord=1 )
            res.append(a_batch_attack)
            gc.collect()
            torch.cuda.empty_cache()
        print("Attribution for batch " + str(i) + " complete")
    
    return res

def run_fgsm(algorithm, model, val_loader, eps, num_batches):
    gc.collect()
    torch.cuda.empty_cache()
                   
    if torch.cuda.is_available():
        model.cuda()
    
    dataiter = iter(val_loader)
    xai = algorithm
    
    res = []
    
    medianAbs_bena = []
    meanAbs_bena = []
    iqr_bena = []
    coef_var_bena =[]
    coef_iqr_bena = []
    
    for i in range(min(num_batches, len(val_loader))):
        images, labels = next(dataiter)
        images = images.to(device)
        labels = labels.to(device)
        for ind in range(len(images)):
            gc.collect()
            torch.cuda.empty_cache()
            input = images[ind].unsqueeze(0)
            images_pgd = fast_gradient_method(model, input, eps, np.inf)
            _, y_pred_pgd = model(images_pgd).max(1)
            a_batch_attack = xai.attribute(inputs=images_pgd, target=y_pred_pgd).sum(axis=1).cpu().detach().numpy()
            meanAbs_bena += compute_mean_abs_dev(a_batch_attack)
            medianAbs_bena += compute_median_abs_dev(a_batch_attack)
            iqr_bena += compute_iqr(a_batch_attack)
            coef_var_bena += compute_coef_var(a_batch_attack)
            coef_iqr_bena += compute_coef_iqr(a_batch_attack)
            gc.collect()
            torch.cuda.empty_cache()
        print("Attribution for batch " + str(i) + " complete")
    
    res.append(medianAbs_bena)
    res.append(meanAbs_bena)
    res.append(iqr_bena)
    res.append(coef_var_bena)
    res.append(coef_iqr_bena)
                   
    print("FGSM Calculation Complete")
    
    return res

def run_fgsm_noise(algorithm, model, val_loader, eps, num_batches, spread):
    gc.collect()
    torch.cuda.empty_cache()
                   
    if torch.cuda.is_available():
        model.cuda()
    
    dataiter = iter(val_loader)
    xai = algorithm
    
    res = []
    
    for i in range(min(num_batches, len(val_loader))):
        images, labels = next(dataiter)
        images = images.to(device)
        labels = labels.to(device)
        for ind in range(len(images)):
            gc.collect()
            torch.cuda.empty_cache()
            input = images[ind].unsqueeze(0)
            images_pgd = fast_gradient_method(model, input, eps, np.inf)
            original = images_pgd
            _, y_original = model(original).max(1)
            attr_original = xai.attribute(original, target=y_original).sum(axis=1).cpu().detach().numpy()

            x = images_pgd.data.cpu().numpy()
            stdev = spread * (np.max(x)-np.min(x))
            noise = np.random.normal(0, stdev, x.shape).astype(np.float32)
            x_plus_noise = x + noise
            x_plus_noise = np.clip(x_plus_noise, 0, 1)
            noisy = torch.from_numpy(x_plus_noise).cpu()
            attr_noisy = xai.attribute(noisy, target=y_original).sum(axis=1).cpu().detach().numpy()
            
            a_batch_attack = np.linalg.norm(attr_noisy.flatten()-attr_original.flatten(),ord=1 )
            res.append(a_batch_attack)
            gc.collect()
            torch.cuda.empty_cache()
        print("Attribution for batch " + str(i) + " complete")
    
    return res

def run_adv2(algorithm, model, adv2np, val_loader, num_batches):
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        model.cuda()
    
    dataiter = iter(val_loader)
    xai = algorithm
    
    res = []
    
    medianAbs_bena = []
    meanAbs_bena = []
    iqr_bena = []
    coef_var_bena =[]
    coef_iqr_bena = []
    
    for i in range(min(num_batches, len(val_loader))):
        images, labels = next(dataiter)
        images = images.to(device)
        labels = labels.to(device)
        for ind in range(len(images)):
            gc.collect()
            torch.cuda.empty_cache()
            input = torch.tensor(adv2np[i]['best_adv'][ind]).unsqueeze(0)
            input = input.to(device)
            a_batch_attack = xai.attribute(inputs=input, target=labels[ind]).sum(axis=1).cpu().detach().numpy()
            meanAbs_bena += compute_mean_abs_dev(a_batch_attack)
            medianAbs_bena += compute_median_abs_dev(a_batch_attack)
            iqr_bena += compute_iqr(a_batch_attack)
            coef_var_bena += compute_coef_var(a_batch_attack)
            coef_iqr_bena += compute_coef_iqr(a_batch_attack)
            gc.collect()
            torch.cuda.empty_cache()
        print("Attribution for batch " + str(i) + " complete")
    
    
    res.append(medianAbs_bena)
    res.append(meanAbs_bena)
    res.append(iqr_bena)
    res.append(coef_var_bena)
    res.append(coef_iqr_bena)
    
    print("ADV2 Calculation Complete")
    
    return res

def run_adv2_noise(algorithm, model, adv2np, val_loader, num_batches, spread):
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        model.cuda()
    
    dataiter = iter(val_loader)
    xai = algorithm
    
    res = []
    
    for i in range(min(num_batches, len(val_loader))):
        images, labels = next(dataiter)
        images = images.to(device)
        labels = labels.to(device)
        for ind in range(len(images)):
            gc.collect()
            torch.cuda.empty_cache()
            input = torch.tensor(adv2np[i]['best_adv'][ind]).unsqueeze(0)
            images_pgd = input.to(device)
            original = images_pgd
            _, y_original = model(original).max(1)
            attr_original = xai.attribute(original, target=y_original).sum(axis=1).cpu().detach().numpy()

            x = images_pgd.data.cpu().numpy()
            stdev = spread * (np.max(x)-np.min(x))
            noise = np.random.normal(0, stdev, x.shape).astype(np.float32)
            x_plus_noise = x + noise
            x_plus_noise = np.clip(x_plus_noise, 0, 1)
            noisy = torch.from_numpy(x_plus_noise).cpu()
            attr_noisy = xai.attribute(noisy, target=y_original).sum(axis=1).cpu().detach().numpy()
            
            a_batch_attack = np.linalg.norm(attr_noisy.flatten()-attr_original.flatten(),ord=1 )
            res.append(a_batch_attack)
            gc.collect()
            torch.cuda.empty_cache()
        print("Attribution for batch " + str(i) + " complete")
    
    print("ADV2 Calculation Complete")
    
    return res


def compute_tpr_fpr(minthres, maxthres, adv, ben):
    tpr = 0
    fpr = 0
    for tempadv, tempben in zip(adv, ben):
        if tempadv >= minthres and tempadv <= maxthres:
            fpr+=1
        if tempben >= minthres and tempben <= maxthres:
            tpr+=1
    return tpr/(len(ben)), fpr/(len(adv))

