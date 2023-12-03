from typing import Any

from tqdm import tqdm

from utils import misc
from utils.draw import draw_tensor
from utils.misc import *


def val(model, dataloader,debug=False):
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
        dict_input['original_size'] = label_val.shape[-2:]
        if debug:
            draw_tensor(label.unsqueeze(dim=0), boxes=labels_box)
        with torch.no_grad():
            # must be batch.
            output = model([dict_input], multimask_output=False)
        # for i in output:
        #     draw_tensor(i["masks"])
        iou = compute_iou(output[0]["masks"], label_val)
        boundary_iou = compute_boundary_iou(output[0]["masks"], label_val)
        # if visualize:
        #     print("visualize")
        #     os.makedirs("./result", exist_ok=True)
        #     masks_hq_vis = (F.interpolate(output[0]["masks"].detach(), (1024, 1024), mode="bilinear",
        #                                   align_corners=False) > 0).cpu()
        #     save_base = os.path.join("./result/" + str(time.time()))
        #     show_iou = torch.tensor([iou.item()])
        #     show_boundary_iou = torch.tensor([boundary_iou.item()])
        #     img = img_val.cpu().numpy().astype(dtype=np.int64)
        #     img = np.transpose(img, (1, 2, 0))
        #     show_anns(masks_hq_vis, None, labels_box.cpu()[0], None, save_base, img, show_iou,
        #               show_boundary_iou)
        iou_res.append(iou)
        boundary_iou_res.append(boundary_iou)
    gather_iou = sum(iou_res) / len(dataloader)
    gather_boundary_iou = sum(boundary_iou_res) / len(dataloader)
    return gather_iou, gather_boundary_iou
