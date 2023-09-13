import torch
from torchsummary import summary
from datasets import load_dataset
from super_image import EdsrModel
from super_image.data import EvalDataset, EvalMetrics
from model_x2 import *
from model import _Residual_Block, _Residual_Block_phase
from PENsr import PENsr
from metrics_calculator import Eval

# Define a function for model evaluation
def evaluate_model(model, dataset, dataset_name):
    print("========================")
    print(f"Evaluation for {dataset_name}")
    print("========================")

    eval_data = EvalDataset(dataset)
    Eval().evaluate(model, eval_data)

def main():
    # Initialize the custom model
    model = Net()
    model.cuda()
    print("==> Loading our model")
    checkpoint = torch.load('checkpoint/model_epoch_1000.pth')
    model.load_state_dict(checkpoint["model"].state_dict())

    # Load and prepare datasets
    set5_dataset = load_dataset('eugenesiow/Set5', 'bicubic_x2', split='validation')
    set14_dataset = load_dataset('eugenesiow/Set14', 'bicubic_x2', split='validation')
    bsd100_dataset = load_dataset('eugenesiow/BSD100', 'bicubic_x2', split='validation')
    urban100_dataset = load_dataset('eugenesiow/Urban100', 'bicubic_x2', split='validation')

    set5_eval = EvalDataset(set5_dataset)
    set14_eval = EvalDataset(set14_dataset)
    bsd100_eval = EvalDataset(bsd100_dataset)
    urban100_eval = EvalDataset(urban100_dataset)

    # Load the pre-trained model
    print("==> Loading the pre-trained model")
    pre_model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)

    # Evaluate the custom model
    evaluate_model(model, set5_dataset, "set5")
    evaluate_model(model, set14_dataset, "set14")
    evaluate_model(model, bsd100_dataset, "bsd100")
    evaluate_model(model, urban100_dataset, "urban100")

    # Evaluate the pre-trained model
    evaluate_model(pre_model, set5_dataset, "set5")
    evaluate_model(pre_model, set14_dataset, "set14")
    evaluate_model(pre_model, bsd100_dataset, "bsd100")
    evaluate_model(pre_model, urban100_dataset, "urban100")

if __name__ == "__main__":
    main()
