import os

import torch.cuda
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from datasets import SegSets
from experiment.eval import box_val
from experiment.train import train
from segment_anything.build_sam_mine import sam_model_registry
from utils.logger import logger_train

# this method is deprecated.
visualize = False
epoch_start = 0
epoch_num = 100
save_freq = 5
checkpoint = "./sam_vit_b_01ec64.pth"

writer = SummaryWriter("./logs")
# set KMP_DUPLICATE_LIB_OK=TRUE
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
optimizer = optim.Adam(
    model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
)
# 余弦退火
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-08)

# LOOP
for epoch in range(epoch_start, epoch_num):
    # EVALUATE
    # if epoch % save_freq == 0 and epoch != 0:
    if epoch % save_freq == 0 and epoch != 0:
        model.eval()
        print(f"Validating...{epoch}")
        iou, boundary_iou = box_val(model, val_dataloader, False)
        writer.add_scalar("iou", iou, epoch)
        writer.add_scalar("boundary_iou", boundary_iou, epoch)
        logger_train.warning(f"{epoch}:{iou}\t{boundary_iou}")
    # TRAIN
    model.train()
    print(f"Training...epoch:{epoch}")
    print("epoch:   ", epoch, "  learning rate:  ", optimizer.param_groups[0]["lr"])
    res_loss = train(model, train_dataloader, optimizer, lr_scheduler)
    writer.add_scalar("training_loss", res_loss, epoch)
    logger_train.info(f"{epoch}:{res_loss}")
    # SAVE
    if epoch % save_freq == 0:
        print(f"Saving pth at './weight_hq_rope/{epoch}.pth'")
        torch.save(model.state_dict(), f"./weight_hq_rope/{epoch}.pth")
