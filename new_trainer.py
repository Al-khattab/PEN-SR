import os
import sys
import copy
from dataclasses import  dataclass
from typing import NamedTuple, Tuple, Union
import random
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from metrics_calcolator import trim, fix_shape
from torch.utils.data.dataloader import DataLoader
from torch.optim import lr_scheduler, Adam
from metrics import AverageMeter, compute_metrics
from torch.utils.tensorboard import SummaryWriter
loss_writer = SummaryWriter('runs/loss_graph')
psnr_writer = SummaryWriter('runs/PSNR_graph')
old_stdout = sys.stdout
from time import time, ctime
t = time()
ctime(t)
file_name = 'logs/EDSR_training_log_'+ctime(t)
log_file = open(file_name,"w+")

class EvalPrediction(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    labels: np.ndarray

def set_seed(seed: int):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
##################################################
@dataclass
class train_args:
    output_dir = 'checkpoint'
    device = 'cuda'
    learning_rate =1e-4
    gamma = 0.5
    num_train_epochs = 1000
    scale = 4
    save_steps = 500
    seed =742
    train_batch_size = 16
    dataloader_num_workers = 16
    dataloader_pin_memory = True
############################################################################
class Trainer:        
    def __init__(self,model,args,train_dataset,test_dataset):
        # Seed must be set before instantiating the model when using model
        self.args = args
        set_seed(self.args.seed)
        if model is None:
            raise RuntimeError("`Trainer` requires a `model`")
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.best_epoch = 0
        self.best_metric = 0.0
#############################################################################
    def train(self,check_point = None):
        # arguments
        args = self.args
        scale = args.scale
        start_epoch = 1
        device = args.device
        num_train_epochs = args.num_train_epochs
        learning_rate = args.learning_rate
        train_batch_size = args.train_batch_size
        train_dataset = self.train_dataset
        train_dataloader = self.get_train_dataloader()
        step_size = int(len(train_dataset) / train_batch_size * 200)

        if check_point != None:
            print("=> loading checkpoint '{}'".format(check_point))
            checkpoint = torch.load(check_point)
            start_epoch = checkpoint["epoch"]+1
            self.model.load_state_dict(checkpoint["model"].state_dict())     
        #optimizer
        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=self.args.gamma)
            
        #training loop
        for epoch in range(start_epoch, num_train_epochs+1):
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * (0.1 ** (epoch // int(num_train_epochs * 0.8)))
            self.model.train()
            epoch_losses = AverageMeter()
            with tqdm(total=(len(train_dataset) - len(train_dataset) % train_batch_size)) as t:
                t.set_description(f'epoch: {epoch}/{num_train_epochs}')
                for data in train_dataloader:
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    preds = self.model(inputs)
                    criterion = nn.L1Loss()
                    loss = criterion(preds, labels)
                    epoch_losses.update(loss.item(), len(inputs))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    torch.cuda.empty_cache()
                    t.set_postfix(loss=f'{epoch_losses.avg:.6f}')
                    t.update(len(inputs))
                    
            loss_writer.add_scalar("Loss/epoch", epoch_losses.avg, epoch)
            sys.stdout = log_file
            print('epoch =',epoch,',','loss =',epoch_losses.avg)
            sys.stdout = old_stdout
            self.save_checkpoint(self.model,epoch)
            self.eval(epoch)
            loss_writer.flush()
        loss_writer.close()
        psnr_writer.close()
        log_file.close()
#####################################################################################
    def save_checkpoint(self,model, epoch):
    	model_folder = "checkpoint/"
    	model_out_path = model_folder + "model_epoch_{}.pth".format(epoch)
    	state = {"epoch": epoch ,"model": model}
    	if not os.path.exists(model_folder):
    		os.makedirs(model_folder)
    	torch.save(state, model_out_path)
    	print("Checkpoint saved to {}".format(model_out_path))
#####################################################################################
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_dataset = self.train_dataset
        return DataLoader(
            dataset=train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
#####################################################################################    
    def eval(self, epoch):
        args = self.args
        device = args.device
        eval_dataloader = self.get_eval_dataloader()
        epoch_psnr = AverageMeter()
        epoch_ssim = AverageMeter()

        self.model.eval()

        for data in eval_dataloader:
            inputs, labels = data
            inputs = trim(inputs)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            

            with torch.no_grad():
                preds = self.model(inputs)
                labels = fix_shape(preds,labels)

            metrics = compute_metrics(EvalPrediction(predictions=preds, labels=labels), scale=args.scale)

            epoch_psnr.update(metrics['psnr'], len(inputs))
            epoch_ssim.update(metrics['ssim'], len(inputs))

        print(f'scale:{str(args.scale)}      eval psnr: {epoch_psnr.avg:.2f}     ssim: {epoch_ssim.avg:.4f}')
        loss_writer.add_scalar("PSNR/epoch", epoch_psnr.avg, epoch)
        psnr_writer.flush()

        if epoch_psnr.avg > self.best_metric:
            self.best_epoch = epoch
            self.best_metric = epoch_psnr.avg

            print(f'best epoch: {epoch}, psnr: {epoch_psnr.avg:.6f}, ssim: {epoch_ssim.avg:.6f}')

#####################################################################################         
    def get_eval_dataloader(self) -> DataLoader:
        eval_dataset = self.test_dataset

        return DataLoader(
            dataset=eval_dataset,
            batch_size=1,
        )
