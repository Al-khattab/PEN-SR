import glob
import cv2
from metrics import AverageMeter, compute_metrics
from typing import NamedTuple, Tuple, Union
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
    
    

################################################
low_path = glob.glob("bi/*.png")
low_img = []

for img in low_path:
    n = cv2.imread(img)
    n = cv2.cvtColor(n,cv2.COLOR_BGR2RGB)
    low_img.append(n)
    
    
high_path = glob.glob("high_res/*.png")
high_img = []
img_name = []
for img in high_path:
    m = cv2.imread(img)
    m = cv2.cvtColor(m,cv2.COLOR_BGR2RGB)
    high_img.append(m)
    img_name.append(img)
    

################################################  
for i in range(len(high_img)):
    print("processing " + img_name[i])
    #change the input to float
    im_l = low_img[i].astype(float)
    im_b = high_img[i].astype(float)
    
    im_l = np.resize(im_l, (im_b.shape[0],im_b.shape[1],im_b.shape[2]))
    #print(im_l.shape)
    #put in shape that can be processesed
    #rem_row = im_l.shape[0]%8
    #rem_col = im_l.shape[1]%8
    #if rem_row != 0:
     #   im_l = im_l[0:im_l.shape[0]-rem_row ,:,:]
    #if rem_col != 0:
    #    im_l = im_l[: ,0:im_l.shape[1]-rem_col,:]
    # change to input tensor form
    im_input = im_l.astype(np.float32).transpose(2,0,1)
    im_input = im_input.reshape(1,im_input.shape[0],im_input.shape[1],im_input.shape[2])
    im_input = Variable(torch.from_numpy(im_input/255.).float())
    ###
    
    
    #krem_row = im_b.shape[0]%8
    #krem_col = im_b.shape[1]%8
    #if krem_row != 0:
     #   im_b = im_b[0:im_b.shape[0]-krem_row ,:,:]
    #if krem_col != 0:
     #   im_b = im_b[: ,0:im_b.shape[1]-krem_col,:]
    im_label = im_b.astype(np.float32).transpose(2,0,1)
    im_label = im_label.reshape(1,im_label.shape[0],im_label.shape[1],im_label.shape[2])
    im_label = Variable(torch.from_numpy(im_label/255.).float())
    ###
    #im_label = fix_shape(im_input,im_label)
    im_input = fix_shape(im_label,im_input)
    metrics = compute_metrics(EvalPrediction(predictions=im_input, labels=im_label), scale= 4)
    print(metrics['psnr'])
    print(metrics['ssim'])

    
    
