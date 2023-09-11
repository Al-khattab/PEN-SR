import argparse
import torch
import cv2
import glob
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

# evalution parameters
cuda    = True
scale  = 4
idx_im = 4
cudnn.benchmark = True

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
#load the model
print("=>Loading the model")
model = Net()
model.cuda()
checkpoint = torch.load('best_PEN.pth')
model.load_state_dict(checkpoint["model"].state_dict())

#load the test images
#im_gt = cv2.imread('0852.png')
#im_gt = cv2.cvtColor(im_gt,cv2.COLOR_BGR2RGB)
im_name = 'patched_0826'
path = 'low_res/'+im_name+'.png'
print(path)
im_l = cv2.imread(path)
im_l = cv2.cvtColor(im_l,cv2.COLOR_BGR2RGB)

model_hr = []
input_lr = []
true_hr = []

    # convert each array into a float type array
#im_gt = im_gt.astype(float)
#true_hr.append((im_gt/255.).astype(np.float32))
im_l = im_l.astype(float)
input_lr.append((im_l/255.).astype(np.float32))
rem_row = im_l.shape[0]%8
rem_col = im_l.shape[1]%8
if rem_row != 0:
    im_l = im_l[0:im_l.shape[0]-rem_row ,:,:]
if rem_col != 0:
    im_l = im_l[: ,0:im_l.shape[1]-rem_col,:]
    #change the shape of the image into a tensor like format format
im_input = im_l.astype(np.float32).transpose(2,0,1)
    #adding extra dimension to put it in the shape that the model can take which is (N,c,w,h)
im_input = im_input.reshape(1,im_input.shape[0],im_input.shape[1],im_input.shape[2])
    #change the input into tensor format with values between (0, 255)
im_input = Variable(torch.from_numpy(im_input/255.).float())


    #use gpu
if cuda:
    with torch.no_grad():
    	model = model.cuda()
    	im_input = im_input.cuda()
else:
    #call the model with cpu
    model = model.cpu()
    im_input = im_input.cpu()
    #pass the input tensor to the model
HR_4x = model(im_input.cuda())
HR_4x = HR_4x.cpu()
im_h = HR_4x.data[0].numpy().astype(np.float32)
im_h = im_h*255.
im_h = np.clip(im_h, 0., 255.)
im_h = im_h.transpose(1,2,0).astype(dtype=np.uint8)
model_hr.append(im_h)
im_h = cv2.cvtColor(im_h,cv2.COLOR_BGR2RGB)
name = im_name + "x4"+".png"
print(name)
cv2.imwrite(name,im_h)
#cv2.imshow(im_name + "x4", im_h)

k = cv2.waitKey(0)

