import os

from torch import optim

from datasets import SegSets
from experiment.eval import *
from segment_anything import sam_model_registry_baseline as sam_model_registry
from utils.logger import logger_train

# this method is deprecated.
visualize = False
epoch_start = 0
epoch_num = 200
save_freq = 5
checkpoint = "./sam_vit_l_0b3195.pth"

# set KMP_DUPLICATE_LIB_OK=TRUE
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("--- init model ---")
# model = sam_model_registry_baseline["vit_b"](checkpoint=checkpoint)
model = sam_model_registry["vit_l"](checkpoint=checkpoint)
print("--- create dataloader ---")
train_dataloader = SegSets.seg_train
val_dataloader = SegSets.seg_val
print(f"--- train:{len(train_dataloader)},val:{len(val_dataloader)} ---")

if torch.cuda.is_available():
    print("--- cuda is available ---")
    model.cuda()
else:
    print("--- 没显卡玩儿个蛋蛋 ---")
    raise Exception("No cuda device.")

for idx, params in enumerate(model.named_parameters()):
    if params[1].requires_grad:
        print(str(idx) + " : " + params[0] + " : " + " need to be trained.")

print("--- create optimizer ---")
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08,
                       weight_decay=0)
# 余弦退火
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-08)

# LOOP
model.eval()
print(f"Validating...")
iou, boundary_iou = point_box_val(model, val_dataloader)
logger_train.warning(f"SAM_VIT_B:{iou}\t{boundary_iou}")
