import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datasets import load_dataset
from super_image.data import EvalDataset, TrainDataset, augment_five_crop
from model_x2 import *
from model import _Residual_Block
from model import _Residual_Block_phase
#from organized_model import EDSR

from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter('runs/Jan15_17-52-23_ankido')
writer = SummaryWriter()
import sys

torch.cuda.empty_cache()
print("===> Loading the training dataset")
augmented_dataset = load_dataset('eugenesiow/Div2k', 'bicubic_x2', split='train')\
    .map(augment_five_crop, batched=True, desc="Augmenting Dataset")
train_dataset = TrainDataset(augmented_dataset)

# Setting up the logger
print("===> Setting the logger")
old_stdout = sys.stdout
from time import time, ctime
t = time()
ctime(t)
file_name = 'logs/EDSR_training_log_'+ctime(t)
log_file = open(file_name,"w+")

print("===> Setting up the training parameters ")
# Training settings
batchSize    = 16     #training batch size
nEpochs      = 1002    #number of epochs to train for
tlr          = 1e-4   #Learning Rate. Default=1e-4
step         = 200    #Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10
cuda         = True  #use cuda
resume       = 'checkpoint/model_epoch_550.pth'     #path to latest checkpoint (default: none)
start_epoch  = 1      #manual epoch number (useful on restarts)
threads      = 16      #number of threads for data loader to use
momentum     = 0.9   #momentum
tweight_decay = float(1e-4)  #weight decay, Default: 0

global model

#check if you can use the gpu 
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found")

#preaparing a seed to randomly inisilize wieghts
seed = random.randint(1, 10000)
print("Random Seed: ", seed)
torch.manual_seed(seed)
if cuda:
    cudnn.benchmark = True    

print("===> Preaparing the dataset")
train_set = train_dataset
training_data_loader = DataLoader(dataset=train_set, num_workers=threads, batch_size= batchSize, shuffle=True)

print("===> Building model")
model = Net()
print("Setting up the criterion")
criterion = nn.L1Loss(size_average=False)
print("===> Setting GPU")

if cuda:
    torch.cuda.empty_cache()
    model = model.cuda()
    criterion = criterion.cuda()

# optionally resume from a checkpoint 
#print("=> loading checkpoint '{}'".format(resume))
checkpoint = torch.load(resume)
start_epoch = checkpoint["epoch"] + 1
model.load_state_dict(checkpoint["model"].state_dict())


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = tlr * (0.1 ** (epoch // step))
    return lr

def train(training_data_loader, optimizer, model, criterion, epoch):
    lr = adjust_learning_rate(optimizer, epoch-1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    running_loss = 0.0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        if cuda:
            input = input.cuda()
            target = target.cuda()
        loss = criterion(model(input), target)
        running_loss =+ loss.item() * input.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        if iteration%50 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.5f}".format(epoch, iteration, len(training_data_loader), loss.item()))
    writer.add_scalar("Loss/train", running_loss, epoch)
    sys.stdout = log_file
    print('epoch =',epoch,',','loss =',running_loss)
    sys.stdout = old_stdout

def save_checkpoint(model, epoch):
    model_folder = "checkpoint/"
    model_out_path = model_folder + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))



print("===> Setting Optimizer")
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr= tlr, weight_decay = tweight_decay , betas = (0.9, 0.999), eps=1e-08)
print("===> Training")
for epoch in range(start_epoch,nEpochs + 1):
    train(training_data_loader, optimizer, model, criterion, epoch)
    if epoch%10 == 0:
        save_checkpoint(model, epoch)
    writer.flush()
print("===> Finalizing the loggers")
writer.close()
log_file.close()
