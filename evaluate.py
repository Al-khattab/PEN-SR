import torch
from torchsummary import summary
from datasets import load_dataset
from super_image import EdsrModel
from metrics_calculator import *
from super_image.data import EvalDataset, EvalMetrics
from model import Net, _Residual_Block, _Residual_Block_phase

def evaluate_model(model, dataset):
    print(f"===> evaluating {dataset.dataset_name}")
    Eval().evaluate(model, dataset)

def main():
    set5_dataset = EvalDataset(load_dataset('eugenesiow/Set5', 'bicubic_x4', split='validation'))
    set14_dataset = EvalDataset(load_dataset('eugenesiow/Set14', 'bicubic_x4', split='validation'))
    bsd100_dataset = EvalDataset(load_dataset('eugenesiow/BSD100', 'bicubic_x4', split='validation'))
    urban100_dataset = EvalDataset(load_dataset('eugenesiow/Urban100', 'bicubic_x4', split='validation'))

    model = Net().cuda()
    pre_model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4).cuda()

    print("========================")
    print("evaluation for our model")
    print("========================")
    evaluate_model(model, set5_dataset)
    evaluate_model(model, set14_dataset)
    evaluate_model(model, bsd100_dataset)
    evaluate_model(model, urban100_dataset)

    print("========================")
    print("evaluation for pre-trained model")
    print("========================")
    evaluate_model(pre_model, set5_dataset)
    evaluate_model(pre_model, set14_dataset)
    evaluate_model(pre_model, bsd100_dataset)
    evaluate_model(pre_model, urban100_dataset)

if __name__ == "__main__":
    main()