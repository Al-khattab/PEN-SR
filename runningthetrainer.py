from new_trainer import Trainer, train_args
from datasets import load_dataset
from super_image.data import TrainDataset, augment_five_crop
from super_image.data import EvalDataset
from PENsr import PENsr

print("===========================================================================================================")
print("loading training dataset")
augmented_dataset = load_dataset('eugenesiow/Div2k', 'bicubic_x4', split='train')\
    .map(augment_five_crop, batched=True, desc="Augmenting Dataset")
train_dataset = TrainDataset(augmented_dataset)

print("loading test dataset")
set5_dataset = load_dataset('eugenesiow/Set5', 'bicubic_x4', split='validation')
set5_eval = EvalDataset(set5_dataset)
print("===========================================================================================================")
model = PENsr()
model.cuda()

trainer = Trainer(
    model=model,                  
    args=train_args,                  
    train_dataset=train_dataset,
    test_dataset = set5_eval,
)
trainer.train() #'checkpoint/model_epoch_721.pth'


