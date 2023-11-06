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
from load_skin import load_skin_data
from load_diabetic import load_diabetic_data

from captum.attr import Saliency
from captum.attr import IntegratedGradients
from captum.attr import GradientShap

# to make the results are reproducible
np.random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed(10)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
       
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd as autograd
                  
from resnet_adv2 import resnet50_soft

def imagenet_resize_postfn(grad):
    grad = grad.abs().max(1, keepdim=True)[0]
    grad = F.avg_pool2d(grad, 4).squeeze(1)
    shape = grad.shape
    grad = grad.view(len(grad), -1)
    grad_min = grad.min(1, keepdim=True)[0]
    grad = grad - grad_min
    grad_max = grad.max(1, keepdim=True)[0]
    grad = grad / torch.max(grad_max, torch.tensor([1e-8], device='cuda'))
    return grad.view(*shape)


def generate_gs_per_batches(model_tup, bx, by, post_fn=None, keep_grad=False):
    model, pre_fn = model_tup[:2]
    bxp = pre_fn(bx)
    logit = model(bxp)
    loss = F.nll_loss(F.log_softmax(logit), by)
    grad = autograd.grad([loss], [bx], create_graph=keep_grad)[0]
    if post_fn is not None:
        grad = post_fn(grad)
    return grad

                  
def freeze_model(model):
    for param in model.parameters():
        param.requires_grad_(False)
                  
                  
def imagenet_normalize(t, mean=None, std=None):
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std= [0.229, 0.224, 0.225]

    ts = []
    for i in range(3):
        ts.append(torch.unsqueeze((t[:, i] - mean[i]) / std[i], 1))
    return torch.cat(ts, dim=1)


def load_model(dataset):
    if dataset=="skin":
        model = torch.load("model/skin/model_skin.pth")
    if dataset == "diabetic":
        model = torch.load("model/diabetic/model_diabetic.pth")
    model_ref = resnet50_soft(True)
    model.to(device)
    model.train(False)
    model_ref.to(device)
    model_ref.train(False)
    freeze_model(model)
    freeze_model(model_ref)
    return (model, imagenet_normalize), (model_ref, imagenet_normalize)


def attack_batch(model_tup, ref_model_tup, bx, by, bm):
    n = len(bx)
    model, pre_fn = model_tup
    ref_model_tup = (ref_model_tup[0], lambda x: x)

    best_dist = np.full((n, ), np.inf, dtype=np.float32)
    best_adv = bx.cpu().numpy()
    # best_adv_gs = np.zeros((n, 224, 244 ), dtype=np.float32)
    best_adv_conf = np.zeros((n,), dtype=np.float32)

    bx0 = bx.clone()
    bx = bx.clone().requires_grad_()
    _, by = by.max(1)
    _, by = by.max(1)
    # print(bx.size())

    for i in range(300):
        bx_p = pre_fn(bx)
        logit = model(bx_p)
        adv_loss = F.nll_loss(F.log_softmax(logit), by, reduction='sum')
        final_grad = autograd.grad([adv_loss], [bx])[0]

        bx.data = bx.data - 1. / 255 * final_grad.sign()
        r = bx.data - bx0
        r.clamp_(-0.031, 0.031)
        bx.data = bx0 + r
        del final_grad
        
        gc.collect()
        torch.cuda.empty_cache()

    bx_adv_start = bx.detach().clone()
    bx = bx_adv_start.clone().requires_grad_()

    for c, num_step in zip([0.001], [101]):
        for i in range(num_step):
            bx_p = pre_fn(bx)
            logit = model(bx_p)
            adv_loss = F.nll_loss(F.log_softmax(logit), by, reduction='sum')
            adv_gs = generate_gs_per_batches(ref_model_tup, bx_p, by, keep_grad=True)

            if i % 10 == 0:
                with torch.no_grad():
                    prob = F.softmax(logit).gather(1, by.view(n, -1)).view(n)
                prob = prob.cpu().numpy()
                now_gs = generate_gs_per_batches(model_tup, bx, by)
                diff = now_gs.detach() - bm
                now_dist = (diff * diff).view(n, -1).sum(1).cpu().numpy()
                mask = np.logical_and(prob > 0.8, now_dist < best_dist)
                indices_np = np.nonzero(mask)[0]
                indices = torch.tensor(indices_np, device=device)
                best_dist[indices_np] = now_dist[indices_np]
                best_adv[indices_np] = bx.detach()[indices].cpu().numpy()
                # best_adv_gs[indices_np] = now_gs.detach()[indices].cpu().numpy()
                best_adv_conf[indices_np] = prob[indices_np]

            diff = adv_gs - bm
            int_loss = (diff * diff).view(n, -1).sum()
            loss = adv_loss + c * int_loss
            final_grad = autograd.grad([loss], [bx])[0]

            bx.data = bx.data - 1./255 * final_grad.sign()
            r = bx.data - bx0
            r.clamp_(-0.031, 0.031)
            bx.data = bx0 + r
            bx.data.clamp_(0, 1)

            if i % 10 == 0:
                succeed_indices = np.nonzero(best_dist < np.inf)[0]

                print('c', c, 'step', i,
                      'succeed:', len(succeed_indices), 'conf:', np.mean(best_adv_conf[succeed_indices]),
                      'dist', np.mean(best_dist[succeed_indices]))

            del final_grad, loss, int_loss
            
            gc.collect()
            torch.cuda.empty_cache()
    return dict(best_dist=best_dist, best_adv=best_adv)



def load_data(model, algorithm, dataset):
    if dataset == "skin":
        train_loader, val_loader = load_skin_data();
    if dataset == "diabetic":
        train_loader, val_loader = load_diabetic_data();
    if algorithm == "grad":
        xai = Saliency(model)
    if algorithm == "ig":
        xai = IntegratedGradients(model)
    res = {
        'img_x': [],
        'img_y': [],
        'benign_gs': []
    }
    dataiter = iter(val_loader)
    for i in range(len(val_loader)):
        gc.collect()
        torch.cuda.empty_cache()
        images, labels = next(dataiter)
        images = images.to(device)
        labels = labels.to(device)
        for ind in range(len(images)):
            inp = images[ind].unsqueeze(0)
            output = model(inp)
            _, out = output.max(1)
            if algorithm == "grad" or algorithm == "ig":
                bm = xai.attribute(inp, target=out).cpu().detach().numpy()
            res['img_x'].append(images[ind].cpu().detach().numpy())
            res['img_y'].append(output.cpu().detach().numpy())
            res['benign_gs'].append(bm)
    name = dataset + algorithm + "_data.npy"
    np.save(name, res)
    return res 
    

def adv2(dataset, algorithm, saved, num_batches):
    model_tup, ref_model_tup = load_model(dataset)
    model, pre_fn = model_tup
    if saved:
        name = dataset + algorithm + "_data.npy"
        dobj = np.load(name, allow_pickle=True).item()
    else:
        dobj = load_data(model, algorithm, dataset)
    img_x, img_y, img_m = dobj['img_x'], dobj['img_y'], dobj['benign_gs']
    batch_size = 16 
    n = len(img_x)
    n_batches = (n + batch_size - 1) 
    results = []
    for i in range(num_batches):
        print(i)
        si = i * batch_size
        ei = min(n, si + batch_size)
        bx_np, by_np, bm_np = img_x[si:ei], img_y[si:ei], img_m[si:ei]
        bx, by, bm = [torch.tensor(arr, device=device) for arr in (bx_np, by_np, bm_np)]
        result = attack_batch(model_tup, ref_model_tup, bx, by, bm)
        # result.update(analyze_batch(model_tup, bx_np, by_np, bm_np, result))
        results.append(result)
    return results