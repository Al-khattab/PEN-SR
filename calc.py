from metrics import AverageMeter, compute_metrics
from typing import NamedTuple, Tuple, Union
import cv2
import argparse
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import time, math, glob
import scipy.io as sio
import matplotlib.pyplot as plt
from metrics_calcolator import *
from model import *
from model import _Residual_Block
from model import _Residual_Block_phase
from metrics_calcolator import trim, fix_shape


class EvalPrediction(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    labels: np.ndarray
name_pred = 'srcnn_results/163085lr_srcnn_x4.png'  
name_label = 'high_res/163085.png'


pred = cv2.imread(name_pred)
pred = cv2.cvtColor(pred,cv2.COLOR_BGR2RGB)

label = cv2.imread(name_label)
label = cv2.cvtColor(label,cv2.COLOR_BGR2RGB)
#pred = cv2.resize(pred, (label.shape[1],label.shape[0]))

print(pred.shape)
print(label.shape)

im_l = pred.astype(float)
rem_row = im_l.shape[0]%8
rem_col = im_l.shape[1]%8

if rem_row != 0:
    im_l = im_l[0:im_l.shape[0]-rem_row ,:,:]
if rem_col != 0:
    im_l = im_l[: ,0:im_l.shape[1]-rem_col,:]
im_input = im_l.astype(np.float32).transpose(2,0,1)
im_input = im_input.reshape(1,im_input.shape[0],im_input.shape[1],im_input.shape[2])
im_input = Variable(torch.from_numpy(im_input/255.).float())


im_b = label.astype(float)
krem_row = im_b.shape[0]%8
krem_col = im_b.shape[1]%8

if krem_row != 0:
    im_b = im_b[0:im_b.shape[0]-krem_row ,:,:]
if krem_col != 0:
    im_b = im_b[: ,0:im_b.shape[1]-krem_col,:]
    
im_label = im_b.astype(np.float32).transpose(2,0,1)
im_label = im_label.reshape(1,im_label.shape[0],im_label.shape[1],im_label.shape[2])
im_label = Variable(torch.from_numpy(im_label/255.).float())

im_label = fix_shape(im_input,im_label)
metrics = compute_metrics(EvalPrediction(predictions=im_input, labels=im_label), scale= 4)

print(metrics['psnr'])
print(metrics['ssim'])
