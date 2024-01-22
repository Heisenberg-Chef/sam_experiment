import os

from torch import optim

from datasets import SegSets
from experiment.eval import *
from segment_anything.build_sam_hq_rope import sam_model_registry
from utils.logger import logger_train

checkpoint = "./sam_vit_b_01ec64.pth"

# set KMP_DUPLICATE_LIB_OK=TRUE
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("--- init model ---")
model = sam_model_registry["vit_b"](checkpoint=checkpoint)
print("--- create dataloader ---")
val_dataloader = SegSets.seg_val

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
# iou, boundary_iou = point_val(model, val_dataloader, k=10)
iou, boundary_iou = box_val(model, val_dataloader)
logger_train.warning(f"SAM_B_HQ_ROPE:{iou}\t{boundary_iou}")
