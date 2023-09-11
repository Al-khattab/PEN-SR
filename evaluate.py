#from PENsr import PENsr
#from model import EDSR
from torchsummary import summary
#datasets
from datasets import load_dataset
#pre_trained edsr
from super_image import EdsrModel
#metrics
from metrics_calcolator import *
from super_image.data import EvalDataset, EvalMetrics

eval_dataset =  EvalDataset(load_dataset('eugenesiow/Set5', 'bicubic_x4', split='validation'))


from model import *
from model import _Residual_Block
from model import _Residual_Block_phase
model = Net()
model.cuda()

#model = PENsr()

#model = EDSR()
model.cuda()
#summary(model, input_size=(3, 64, 64))
print("==> Loading our model")
checkpoint = torch.load('best_PEN.pth')
model.load_state_dict(checkpoint["model"].state_dict())


set5_dataset = load_dataset('eugenesiow/Set5', 'bicubic_x4', split='validation')
#DIV2K_dataset = load_dataset('eugenesiow/Div2k', 'bicubic_x4', split='validation')
set14_dataset = load_dataset('eugenesiow/Set14', 'bicubic_x4', split='validation')
bsd100_dataset = load_dataset('eugenesiow/BSD100', 'bicubic_x4', split='validation')
urban100_dataset = load_dataset('eugenesiow/Urban100', 'bicubic_x4', split='validation')

set5_eval = EvalDataset(set5_dataset)
#DIV2K_eval = EvalDataset(DIV2K_dataset)
set14_eval = EvalDataset(set14_dataset)
bsd100_eval = EvalDataset(bsd100_dataset)
urban100_eval = EvalDataset(urban100_dataset)

print("==> Loading the pre-trained model")
pre_model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4)
print("========================")
print("evaluation for our model")
print("========================")
print("===> evaluating DIV2K")
#Eval().evaluate(model, DIV2K_eval)
print("===> evaluating set5")
Eval().evaluate(model, set5_eval)
print("===> evaluating set14")
Eval().evaluate(model, set14_eval)
print("===> evaluating bsd100")
Eval().evaluate(model, bsd100_eval)
print("===> evaluating urban100")
Eval().evaluate(model, urban100_eval)

print("========================")
print("evaluation for pre trained model")
print("========================")

print("===> evaluating set5")
Eval().evaluate(pre_model, set5_eval)
print("===> evaluating set14")
Eval().evaluate(pre_model, set14_eval)
print("===> evaluating bsd100")
Eval().evaluate(pre_model, bsd100_eval)
print("===> evaluating urban100")
Eval().evaluate(pre_model, urban100_eval)
