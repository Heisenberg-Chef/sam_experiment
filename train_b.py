import os

import torch.cuda
from torch import optim

from datasets import SegSets
from experiment.eval import val
from experiment.train import train
from segment_anything.build_sam_hq import sam_model_registry
from utils.logger import logger_train

# this method is deprecated.
visualize = False
epoch_start = 0
epoch_num = 200
save_freq = 5
checkpoint = "./sam_hq_vit_b.pth"

# set KMP_DUPLICATE_LIB_OK=TRUE
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("--- init model ---")
model = sam_model_registry["vit_b"](checkpoint=checkpoint)

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
for epoch in range(epoch_start, epoch_num):
    # EVALUATE
    # if epoch % save_freq == 0 and epoch != 0:
    if epoch % save_freq == 0:
        model.eval()
        print(f"Validating...{epoch}")
        iou, boundary_iou = val(model, val_dataloader, True)
        logger_train.warn(f"{epoch}:{iou}\t{boundary_iou}")
    # TRAIN
    model.train()
    print(f"Training...epoch:{epoch}")
    print("epoch:   ", epoch, "  learning rate:  ", optimizer.param_groups[0]["lr"])
    res_loss = train(model, train_dataloader, optimizer, lr_scheduler)
    logger_train.info(f"{epoch}:{res_loss}")
    # SAVE
    if epoch % save_freq == 0:
        print(f"Saving pth at './weights_b/{epoch}.pth'")
        torch.save(model.state_dict(), f"./weights_b/{epoch}.pth")
