import random

import torch.cuda
import torch.nn.functional as F
from tqdm import tqdm

from utils import misc
from utils.Loss import loss_masks


def train(model, dataloader, optimizer, lr_scheduler, input_keys=['box']):
    loss_value = 0
    for idx, data in enumerate(tqdm(dataloader)):
        inputs, labels = data['image'], data['label']
        inputs = inputs.cuda()
        labels = labels.cuda()

        # input prompt
        # input_keys = ['box', 'point', 'noise_mask']
        input_keys = input_keys
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
            dict_input['original_size'] = inputs[b_i].shape[-2:]
            batched_input.append(dict_input)

        batched_output = model(batched_input, multimask_output=False)
        batch_len = len(batched_output)
        masks = torch.cat([batched_output[i_l]["masks"] for i_l in range(batch_len)])
        # src
        loss_mask, loss_dice,loss_hausdorff = loss_masks(masks, labels / 255.0, len(masks))
        loss = loss_mask + loss_dice + loss_hausdorff
        loss_dict = {"loss_mask": loss_mask.item(), "loss_dice": loss_dice.item(),"loss_hausdorff":loss_hausdorff.item()}
        losses_reduced_scaled = sum(loss_dict.values())
        loss_value += losses_reduced_scaled
        # 优化器更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    lr_scheduler.step()
    loss_value = loss_value / len(dataloader)
    return loss_value
