from typing import Any

# from scipy.ndimage.morphology import distance_transform_edt as edt
from tqdm import tqdm

# from scipy.ndimage.morphology import distance_transform_edt as edt
from utils import misc
from utils.draw import draw_tensor
from utils.misc import *


def box_val(model, dataloader, debug=False):
    iou_res = []
    boundary_iou_res = []
    for data in tqdm(dataloader):
        input_, label, label_val, img_val = data['image'], data['label'], data['ori_label'], data['ori_img']
        input_ = input_.squeeze(dim=0).cuda()
        label = label.squeeze(dim=0).cuda()
        label_val = label_val.cuda()
        img_val = img_val.squeeze(dim=0).cuda()
        # BATCH SIZE MUST EQUAL ONE.
        dict_input: dict[str, Any] = {}
        labels_box = misc.masks_to_boxes(label)
        dict_input['boxes'] = labels_box
        # labels_points = misc.masks_sample_points(label[:, 0, :, :])
        dict_input['image'] = torch.as_tensor(input_, device=model.device).contiguous()
        dict_input['original_size'] = label.shape[-2:]
        with torch.no_grad():
            # must be batch.
            output = model([dict_input], multimask_output=False)
        if debug:
            draw_tensor(label.unsqueeze(dim=0), boxes=labels_box)
            draw_tensor(output[0]["masks"], boxes=labels_box)
        iou = compute_iou(output[0]["masks"], label.unsqueeze(dim=0))
        boundary_iou = compute_boundary_iou(output[0]["masks"], label.unsqueeze(dim=0))
        iou_res.append(iou)
        boundary_iou_res.append(boundary_iou)
    gather_iou = sum(iou_res) / len(dataloader)
    gather_boundary_iou = sum(boundary_iou_res) / len(dataloader)
    return gather_iou, gather_boundary_iou


def point_val(model, dataloader, k=10, debug=False):
    iou_res = []
    boundary_iou_res = []
    for data in tqdm(dataloader):
        input_, label, label_val, img_val = data['image'], data['label'], data['ori_label'], data['ori_img']
        input_ = input_.squeeze(dim=0).cuda()
        label = label.squeeze(dim=0).cuda()
        label_val = label_val.cuda()
        img_val = img_val.squeeze(dim=0).cuda()
        # BATCH SIZE MUST EQUAL ONE.
        dict_input: dict[str, Any] = {}
        point_coords = misc.masks_sample_points(label, k=k)
        dict_input['image'] = torch.as_tensor(input_, device=model.device).contiguous()
        dict_input['original_size'] = label.shape[-2:]
        dict_input['point_coords'] = point_coords
        dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None, :]
        with torch.no_grad():
            # must be batch.
            output = model([dict_input], multimask_output=False)
        iou = compute_iou(output[0]["masks"], label.unsqueeze(dim=0))
        boundary_iou = compute_boundary_iou(output[0]["masks"], label.unsqueeze(dim=0))
        iou_res.append(iou)
        boundary_iou_res.append(boundary_iou)
    gather_iou = sum(iou_res) / len(dataloader)
    gather_boundary_iou = sum(boundary_iou_res) / len(dataloader)
    return gather_iou, gather_boundary_iou


def text_val(model, dataloader, clip, text=""):
    iou_res = []
    boundary_iou_res = []
    for data in tqdm(dataloader):
        input_, label, label_val, img_val = data['image'], data['label'], data['ori_label'], data['ori_img']
        input_ = input_.squeeze(dim=0).cuda()
        label = label.squeeze(dim=0).cuda()
        label_val = label_val.cuda()
        img_val = img_val.squeeze(dim=0).cuda()
        # BATCH SIZE MUST EQUAL ONE.
        dict_input: dict[str, Any] = {}
        point_coords = misc.masks_sample_points(label, k=k)
        dict_input['image'] = torch.as_tensor(input_, device=model.device).contiguous()
        dict_input['original_size'] = label.shape[-2:]
        text_encode = clip(text)
        dict_input['text_encode'] = text_encode
        with torch.no_grad():
            # must be batch.
            output = model([dict_input], multimask_output=False)
        iou = compute_iou(output[0]["masks"], label.unsqueeze(dim=0))
        boundary_iou = compute_boundary_iou(output[0]["masks"], label.unsqueeze(dim=0))
        iou_res.append(iou)
        boundary_iou_res.append(boundary_iou)
    gather_iou = sum(iou_res) / len(dataloader)
    gather_boundary_iou = sum(boundary_iou_res) / len(dataloader)
    return gather_iou, gather_boundary_iou


def point_box_val(model, dataloader, k=1, debug=False):
    iou_res = []
    boundary_iou_res = []
    for data in tqdm(dataloader):
        input_, label, label_val, img_val = data['image'], data['label'], data['ori_label'], data['ori_img']
        input_ = input_.squeeze(dim=0).cuda()
        label = label.squeeze(dim=0).cuda()
        label_val = label_val.cuda()
        img_val = img_val.squeeze(dim=0).cuda()
        # BATCH SIZE MUST EQUAL ONE.
        dict_input: dict[str, Any] = {}
        point_coords = misc.masks_sample_points(label, k=k)
        dict_input['image'] = torch.as_tensor(input_, device=model.device).contiguous()
        dict_input['original_size'] = label.shape[-2:]
        dict_input['point_coords'] = point_coords
        dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None, :]
        labels_box = misc.masks_to_boxes(label)
        dict_input['boxes'] = labels_box
        with torch.no_grad():
            # must be batch.
            output = model([dict_input], multimask_output=False)
        iou = compute_iou(output[0]["masks"], label.unsqueeze(dim=0))
        boundary_iou = compute_boundary_iou(output[0]["masks"], label.unsqueeze(dim=0))
        iou_res.append(iou)
        boundary_iou_res.append(boundary_iou)
    gather_iou = sum(iou_res) / len(dataloader)
    gather_boundary_iou = sum(boundary_iou_res) / len(dataloader)
    return gather_iou, gather_boundary_iou
