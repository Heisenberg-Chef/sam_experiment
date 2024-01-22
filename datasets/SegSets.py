import glob

import torch
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.transform import *


class SegSets(Dataset):
    def __init__(self, data_path=r"E:\datasets\shit\**\train", transform=transforms.Compose([RandomHFlip(),
                                                                                             LargeScaleJitter()]),
                 evaluation=False):
        self.data_list = [i.rstrip("_mask.png") for i in glob.glob(data_path + "\\" + "*_mask.png")]
        self.datasets = {"image": [], "mask": []}
        self.transform = transform
        self.evaluation = evaluation
        for i in self.data_list:
            self.datasets["image"].append(i + ".png")
            self.datasets["mask"].append(i + "_mask.png")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path = self.datasets["image"][index]
        mask_path = self.datasets["mask"][index]
        img = io.imread(image_path)
        gt = io.imread(mask_path)

        im = torch.tensor(img.copy(), dtype=torch.float32)

        im = torch.transpose(torch.transpose(im, 1, 2), 0, 1)
        gt = torch.unsqueeze(torch.tensor(gt, dtype=torch.float32), 0)

        sample = {
            "index": torch.from_numpy(np.array(index)),
            "image": im,
            "label": gt,
            "shape": torch.tensor(im.shape[-2:])
        }

        if self.transform:
            sample = self.transform(sample)

        if self.evaluation:
            sample["ori_label"] = gt.type(torch.int32)
            sample["ori_img"] = im.type(torch.int32)
        sample["name"] = self.datasets["image"][index]

        return sample


seg_train = DataLoader(SegSets(data_path=r"E:\datasets\seg\**\train"), batch_size=5, shuffle=True)
# evaluation data must batch size equals one.
seg_val = DataLoader(SegSets(data_path=r"E:\datasets\seg\**\val", evaluation=True), batch_size=1)
