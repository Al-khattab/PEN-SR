from tqdm.auto import tqdm
import torch
from torch import nn
import random
import numpy as np
from typing import NamedTuple, Tuple, Union
from torch.utils.data import DataLoader
from metrics import *

class EvalPrediction(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    labels: np.ndarray


def trim(tensor):
    rem_row = tensor.shape[2]%8
    rem_col = tensor.shape[3]%8
    if rem_row != 0:
        tensor = tensor[:,:,0:tensor.shape[2]-rem_row,:]
    if rem_col != 0:
        tensor = tensor[: ,:,:,0:tensor.shape[3]-rem_col]
    return tensor

def fix_shape(pred, label):
    row_diff = label.shape[2] - pred.shape[2]
    col_diff = label.shape[3] - pred.shape[3]
    if row_diff != 0:
        label = label[:,:,0:label.shape[2]-row_diff,:]
    if col_diff != 0:
        label = label[:,:,:,0:label.shape[3]-col_diff]
    return label

class Eval:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def evaluate(self, model: nn.Module, dataset: Dataset, scale: int = None):
        if scale is None:
            if len(dataset) > 0:
                scale = get_scale_from_dataset(dataset)
            else:
                raise ValueError(f"Unable to calculate scale from empty dataset.")

        eval_dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=16)
        epoch_psnr = AverageMeter()
        epoch_ssim = AverageMeter()
        for i, data in tqdm(enumerate(eval_dataloader), total=len(dataset), desc='Evaluating dataset'):
            inputs, labels = data
            inputs = trim(inputs)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            model.to(self.device)
            with torch.no_grad():
                preds = model(inputs)
                labels = fix_shape(preds,labels)

            metrics = compute_metrics(EvalPrediction(predictions=preds, labels=labels), scale=scale)

            epoch_psnr.update(metrics['psnr'], len(inputs))
            epoch_ssim.update(metrics['ssim'], len(inputs))
        print(f'scale:{str(scale)}      eval psnr: {epoch_psnr.avg:.2f}     ssim: {epoch_ssim.avg:.4f}')
