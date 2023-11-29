import random
import os
import time
from typing import Any
import numpy as np

import torch.cuda
import torch.nn.functional as F
import tqdm
from matplotlib import pyplot as plt
from torch import optim

from datasets import ShitSets
from segment_anything.build_sam_hq import sam_model_registry
from utils import misc
from utils.Loss import loss_masks
from utils.draw import show_anns
from utils.logger import logger_train, logger_val
from utils.misc import compute_iou, compute_boundary_iou

# this method is deprecated.
visualize = False
epoch_start = 1
epoch_num = 200
save_freq = 5
checkpoint = "./sam_hq_vit_l.pth"

# set KMP_DUPLICATE_LIB_OK=TRUE
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("--- init model ---")
model = sam_model_registry["vit_l"](checkpoint=checkpoint)

print("--- create dataloader ---")
train_dataloader = ShitSets.shit_train
val_dataloader = ShitSets.shit_val
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
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

# LOOP
for epoch in range(epoch_start, epoch_num):
    iou_res = []
    boundary_iou_res = []
    # EVALUATE
    if epoch % save_freq == 0 and epoch != 0:
        model.eval()
        print("Validating...")
        for data in tqdm.tqdm(val_dataloader):
            input_, label, label_val, img_val = data['image'], data['label'], data['ori_label'], data['ori_img']
            input_ = input_.squeeze(dim=0).cuda()
            label = label.squeeze(dim=0).cuda()
            label_val = label_val.cuda()
            img_val = img_val.squeeze(dim=0).cuda()
            # BATCH SIZE MUST EQUAL ONE.
            dict_input: dict[str, Any] = {}
            labels_box = misc.masks_to_boxes(label_val[:, 0, :, :])
            dict_input['image'] = torch.as_tensor(input_, device=model.device).contiguous()
            dict_input['boxes'] = labels_box
            dict_input['original_size'] = label_val.shape[-2:]

            with torch.no_grad():
                output = model([dict_input], multimask_output=False)

            iou = compute_iou(output[0]["masks"], label_val)
            boundary_iou = compute_boundary_iou(output[0]["masks"], label_val)

        # test
        # mask_numpy = output[0]["masks"][0].permute(1, 2, 0).cpu().numpy()
        #
        # # 显示图像
        # plt.imshow(mask_numpy, cmap='gray')
        # plt.show()

        if visualize:
            print("visualize")
            os.makedirs("./result", exist_ok=True)
            masks_hq_vis = (F.interpolate(output[0]["masks"].detach(), (1024, 1024), mode="bilinear",
                                          align_corners=False) > 0).cpu()
            save_base = os.path.join("./result/" + str(time.time()))
            show_iou = torch.tensor([iou.item()])
            show_boundary_iou = torch.tensor([boundary_iou.item()])
            img = img_val.cpu().numpy().astype(dtype=np.int64)
            img = np.transpose(img, (1, 2, 0))
            show_anns(masks_hq_vis, None, labels_box.cpu()[0], None, save_base, img, show_iou,
                      show_boundary_iou)
        iou_res.append(iou)
        boundary_iou_res.append(boundary_iou)
        gather_iou = sum(iou_res)
        gather_boundary_iou = sum(boundary_iou_res)
        logger_train.info(f"{epoch}:{gather_iou},{gather_boundary_iou}")

    # TRAIN
    loss_value = 0
    model.train()

    print(f"Training...epoch:{epoch}")
    print("epoch:   ", epoch, "  learning rate:  ", optimizer.param_groups[0]["lr"])
    for idx, data in enumerate(tqdm.tqdm(train_dataloader)):
        inputs, labels = data['image'], data['label']
        inputs = inputs.cuda()
        labels = labels.cuda()

        # input prompt
        input_keys = ['box', 'point', 'noise_mask']
        # mask --> boxes ，通过mask找到框
        labels_box = misc.masks_to_boxes(labels[:, 0, :, :])
        # 找到点
        try:
            labels_points = misc.masks_sample_points(labels[:, 0, :, :])
        except:
            # less than 10 points
            input_keys = ['box', 'noise_mask']

        labels_256 = F.interpolate(labels, size=(256, 256), mode='bilinear')
        labels_noisemask = misc.masks_noise(labels_256)
        batched_input = []
        for b_i in range(len(inputs)):
            dict_input = {}
            input_image = torch.as_tensor(inputs[b_i], device=model.device).contiguous()
            # 每一次训练随机生成一个prompts，
            dict_input['image'] = input_image
            input_type = random.choice(input_keys)

            if input_type == 'box':
                dict_input['boxes'] = labels_box[b_i:b_i + 1]
            elif input_type == 'point':
                point_coords = labels_points[b_i:b_i + 1]  # try except
                dict_input['point_coords'] = point_coords
                dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None, :]
            elif input_type == 'noise_mask':
                dict_input['mask_inputs'] = labels_noisemask[b_i:b_i + 1]
            else:
                raise NotImplementedError
            # dict_input['original_size'] = original_size[b_i] 训练过程中不进行尺寸缩放，方便GT进行比较
            # torch.Size([1024, 1024])
            dict_input['original_size'] = inputs[b_i].shape[-2:]
            batched_input.append(dict_input)

        # print(batched_input)

        batched_output = model(batched_input, multimask_output=False)
        batch_len = len(batched_output)
        masks = torch.cat([batched_output[i_l]["masks"] for i_l in range(batch_len)])
        # src
        loss_mask, loss_dice = loss_masks(masks, labels / 255.0, len(masks))
        loss = loss_mask + loss_dice
        loss_dict = {"loss_mask": loss_mask.item(), "loss_dice": loss_dice.item()}
        losses_reduced_scaled = sum(loss_dict.values())
        loss_value += losses_reduced_scaled
        # 优化器更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(f"REPORT AT {epoch}:")
        # for name in data["name"]:
        #     print("\t" + name)
        # print(f"loss value:{loss_value / (idx + 1)}")
    print("Finished epoch:", epoch)
    print("Averaged stats:", loss_value / len(train_dataloader))
    logger_train.info(f"{epoch}:{loss_value / len(train_dataloader)}")
    lr_scheduler.step()

    # SAVE
    if epoch % save_freq == 0 and epoch != 0:
        print(f"Saving pth at './weights/{epoch}.pth'")
        torch.save(model.state_dict(), f"./weights/{epoch}.pth")
