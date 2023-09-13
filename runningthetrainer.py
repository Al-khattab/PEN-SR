import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from super_image.data import TrainDataset, augment_five_crop, EvalDataset
from PENsr import PENsr
from new_trainer import Trainer, train_args

# Load training dataset and augment it
print("Loading training dataset...")
train_dataset = TrainDataset(
    load_dataset('eugenesiow/Div2k', 'bicubic_x4', split='train')
    .map(augment_five_crop, batched=True, desc="Augmenting Dataset")
)

# Load test dataset
print("Loading test dataset...")
set5_eval = EvalDataset(
    load_dataset('eugenesiow/Set5', 'bicubic_x4', split='validation')
)

# Initialize and move model to GPU
model = PENsr().cuda()

# Create DataLoader for training dataset
train_loader = DataLoader(train_dataset, batch_size=train_args.per_device_train_batch_size, shuffle=True)

# Create DataLoader for test dataset
test_loader = DataLoader(set5_eval, batch_size=train_args.per_device_eval_batch_size)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataloader=train_loader,
    eval_dataloader=test_loader,
)

# Start training
print("Training...")
trainer.train()