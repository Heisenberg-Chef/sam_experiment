import torch
import torch.nn.functional as F
import numpy as np
import cv2


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)
    y = y.to(masks)
    x = x.to(masks)

    x_mask = ((masks > 128) * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks > 128), 1e8).flatten(1).min(-1)[0]

    y_mask = ((masks > 128) * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks > 128), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def masks_sample_points(masks, k=10):
    """Sample points on mask
    """
    if masks.numel() == 0:
        return torch.zeros((0, 2), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)
    y = y.to(masks)
    x = x.to(masks)

    # k = 10
    samples = []
    for b_i in range(len(masks)):
        select_mask = (masks[b_i] > 128)
        # 通过给定的掩码，选择符合条件的
        x_idx = torch.masked_select(x, select_mask)
        y_idx = torch.masked_select(y, select_mask)
        # 随机生成 0 - (n-1) 的张量
        perm = torch.randperm(x_idx.size(0))
        idx = perm[:k]  # 取前10个
        samples_x = x_idx[idx]
        samples_y = y_idx[idx]
        samples_xy = torch.cat((samples_x[:, None], samples_y[:, None]), dim=1)
        samples.append(samples_xy)
    # 根据mask生成10个随机点
    samples = torch.stack(samples)
    return samples


def masks_noise(masks):
    def get_incoherent_mask(input_masks, sfact):
        mask = input_masks.float()
        w = input_masks.shape[-1]
        h = input_masks.shape[-2]
        mask_small = F.interpolate(mask, (h // sfact, w // sfact), mode='bilinear')
        mask_recover = F.interpolate(mask_small, (h, w), mode='bilinear')
        mask_residue = (mask - mask_recover).abs()
        mask_residue = (mask_residue >= 0.01).float()
        return mask_residue

    gt_masks_vector = masks / 255
    mask_noise = torch.randn(gt_masks_vector.shape, device=gt_masks_vector.device) * 1.0
    inc_masks = get_incoherent_mask(gt_masks_vector, 8)
    gt_masks_vector = ((gt_masks_vector + mask_noise * inc_masks) > 0.5).float()
    gt_masks_vector = gt_masks_vector * 255

    return gt_masks_vector


def mask_iou(pred_label, label):
    '''
    calculate mask iou for pred_label and gt_label
    '''

    pred_label = (pred_label > 0)[0].int()
    label = (label > 128)[0].int()

    intersection = ((label * pred_label) > 0).sum()
    union = ((label + pred_label) > 0).sum()
    return intersection / union


def compute_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if (preds.shape[2] != target.shape[2] or preds.shape[3] != target.shape[3]):  # 缩放到合适的比例
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0, len(preds)):
        iou = iou + mask_iou(postprocess_preds[i], target[i])
    return iou / len(preds)


def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1: h + 1, 1: w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def boundary_iou(gt, dt, dilation_ratio=0.02):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    device = gt.device
    dt = (dt > 0)[0].cpu().byte().numpy()
    gt = (gt > 128)[0].cpu().byte().numpy()

    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    boundary_iou = intersection / union
    return torch.tensor(boundary_iou).float().to(device)


def compute_boundary_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if (preds.shape[2] != target.shape[2] or preds.shape[3] != target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0, len(preds)):
        iou = iou + boundary_iou(target[i], postprocess_preds[i])
    return iou / len(preds)

