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

def load_and_preprocess_images(directory):
    image_paths = glob.glob(directory + "/*.png")
    images = []
    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image.astype(float))
    return images

def process_image(im_l, im_b):
    # Add any necessary preprocessing here
    im_input = im_l.astype(np.float32).transpose(2, 0, 1)
    im_input = im_input.reshape(1, im_input.shape[0], im_input.shape[1], im_input.shape[2])
    im_input = Variable(torch.from_numpy(im_input / 255.).float())

    im_label = im_b.astype(np.float32).transpose(2, 0, 1)
    im_label = im_label.reshape(1, im_label.shape[0], im_label.shape[1], im_label.shape[2])
    im_label = Variable(torch.from_numpy(im_label / 255.).float())

    im_input = fix_shape(im_label, im_input)
    
    metrics = compute_metrics(EvalPrediction(predictions=im_input, labels=im_label), scale=4)
    return metrics

if __name__ == "__main__":
    low_img = load_and_preprocess_images("bi")
    high_img = load_and_preprocess_images("high_res")

    for i, img_name in enumerate(high_img):
        print("Processing " + img_name)
        metrics = process_image(low_img[i], high_img[i])
        print(metrics['psnr'])
        print(metrics['ssim'])
    
    
