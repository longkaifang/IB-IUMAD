from __future__ import division

import json
import logging

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from datasets.base_dataset import BaseDataset, TestBaseTransform, TrainBaseTransform
from datasets.image_reader import build_image_reader,build_image_reader1
from datasets.transforms import RandomColorJitter

# class_label_mapping = {
#     'bottle': 0,
#     'cable': 1,
#     'capsule': 2,
#     'carpet': 3,
#     'grid': 4,
#     'hazelnut': 5,
#     'leather': 6,
#     'metal_nut': 7,
#     'pill': 8,
#     'screw': 9,
#     'tile': 10,
#     'toothbrush': 11,
#     'transistor': 12,
#     'wood': 13,
#     'zipper': 14
# }

# class_label_mapping = {
#     'candle': 0,
#     'capsules': 1,
#     'cashew': 2,
#     'chewinggum': 3,
#     'fryum': 4,
#     'macaroni1': 5,
#     'macaroni2': 6,
#     'pcb1': 7,
#     'pcb2': 8,
#     'pcb3': 9,
#     'pcb4': 10,
#     'pipe_fryum': 11,
# }

class_label_mapping = {
    'bagel': 0,
    'cable_gland': 1,
    'carrot': 2,
    'cookie': 3,
    'dowel': 4,
    'foam': 5,
    'peach': 6,
    'potato': 7,
    'rope': 8,
    'tire': 9,
}

# CandyCane ChocolateCookie ChocolatePraline Confetto GummyBear HazelnutTruffle LicoriceSandwich Lollipop Marshmallow PeppermintCandy
# class_label_mapping = {
#     'CandyCane': 0,
#     'ChocolateCookie': 1,
#     'ChocolatePraline': 2,
#     'Confetto': 3,
#     'GummyBear': 4,
#     'HazelnutTruffle': 5,
#     'LicoriceSandwich': 6,
#     'Lollipop': 7,
#     'Marshmallow': 8,
#     'PeppermintCandy': 9,
# }

logger = logging.getLogger("global_logger")

def build_custom_dataloader(cfg, training, distributed=False): # 加载数据

    image_reader = build_image_reader(cfg.image_reader)
    image_reader_xyz = build_image_reader1(cfg.image_reader)

    normalize_fn = transforms.Normalize(mean=cfg["pixel_mean"], std=cfg["pixel_std"])
    if training:
        transform_fn = TrainBaseTransform(
            cfg["input_size"], cfg["hflip"], cfg["vflip"], cfg["rotate"]
        )
    else:
        transform_fn = TestBaseTransform(cfg["input_size"])

    colorjitter_fn = None
    if cfg.get("colorjitter", None) and training:
        colorjitter_fn = RandomColorJitter.from_params(cfg["colorjitter"])

    logger.info("building CustomDataset from: {}".format(cfg["meta_file"]))

    dataset = CustomDataset(
        image_reader,
        image_reader_xyz,
        cfg["meta_file"],
        training,
        transform_fn=transform_fn,
        normalize_fn=normalize_fn,
        colorjitter_fn=colorjitter_fn,
    )

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)

    if training:
        data_loader = DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=cfg["workers"],
            pin_memory=False,
            sampler=sampler,
            drop_last=True
        )
    else:
        data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=cfg["workers"],
        pin_memory=False,
        sampler=sampler,
        drop_last=True
        )

    return data_loader


class CustomDataset(BaseDataset):
    def __init__(
        self,
        image_reader,
        image_reader_xyz,
        meta_file,
        training,
        transform_fn,
        normalize_fn,
        colorjitter_fn=None,
    ):
        self.image_reader = image_reader
        self.image_reader_xyz =image_reader_xyz
        self.meta_file = meta_file
        self.training = training
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        self.colorjitter_fn = colorjitter_fn

        # construct metas
        with open(meta_file, "r") as f_r: # meta_file--/home/admin1/2Tsdb/lkf/uniform-3dad/dataset/MVTec_3DAD/split/9_train.json 
            self.metas = []
            for line in f_r:
                meta = json.loads(line)
                self.metas.append(meta)
            # self.metas----[{'filename': 'bagel/train/good/rgb/221.png', 'label': 0, 'label_name': 'good'},{...},...]

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index): #index的取值是 len(self.metas)-1
        input = {}
        meta = self.metas[index] #在每次迭代时，DataLoader 会根据 batch_size 和 shuffle 参数生成一组 index，
                                 #并调用 __getitem__ 方法获取对应的数据样本。

        # read image
        filename = meta["filename"] # potato/validation/good/rgb/000.png
        filename_xyz = meta["filename_xyz"] # potato/validation/good/xyz/000.tiff
        label = meta["label"]
        image = self.image_reader(meta["filename"]) #读取RBG图片
        image_xyz = self.image_reader_xyz(meta["filename_xyz"]) #读取xyz图片
        # print("*image**",image.shape[0],image.shape[1])
        # print("*image_xyz**",image.shape[0],image.shape[1])
        input.update(
            {
                "filename": filename,
                "filename_xyz": filename_xyz,
                "height": image.shape[0],
                "width": image.shape[1],
                "label": label,
            }
        ) # 更新输入尺寸

        if meta.get("clsname", None):
            input["clsname"] = meta["clsname"]
        else:
            # input["clsname"] = filename.split("/")[-4] #原来
            input["clsname"] = filename.split("/")[0]  #Mvtec 3D AD  RGB
            # print("1111",filename)
        
        # print("Class Name:",input["clsname"])
        one_hot_label = np.eye(len(class_label_mapping))[class_label_mapping[input["clsname"]]]
        # print(one_hot_label)
        # input.update(
        #     {
        #         "clslabel": one_hot_label
        #     }
        # )
        
        image = Image.fromarray(image, "RGB") #原来 # 将一个 NumPy 数组 (image) 转换为 PIL（Python Imaging Library）图像对象
        image_xyz = Image.fromarray(image_xyz, "RGB")
        # image = Image.fromarray(image, "L")
        # img = np.array(pil_img)

        # read / generate mask
        if meta.get("maskname", None): #none
            mask = self.image_reader(meta["maskname"], is_mask=True)
        else:
            if label == 0:  # good
                mask = np.zeros((image.height, image.width)).astype(np.uint8) #创建一个与输入图像 (image) 尺寸相同的全零矩阵
            elif label == 1:  # defective
                mask = (np.ones((image.height, image.width)) * 255).astype(np.uint8) #初始时，掩码的所有像素值都为 255，表示所有区域都被标记。
            else:
                raise ValueError("Labels must be [None, 0, 1]!")

        mask = Image.fromarray(mask, "L") #这行代码的作用是将一个 NumPy 数组 (mask) 转换为 PIL（Python Imaging Library）图像对象，
                                         #并指定图像的格式为 "L"（灰度图像）

        if self.transform_fn:
            image, mask = self.transform_fn(image, mask)
            image_xyz, mask_xyz = self.transform_fn(image_xyz, mask)
        if self.colorjitter_fn:
            image = self.colorjitter_fn(image)
            image_xyz = self.colorjitter_fn(image_xyz)
        image = transforms.ToTensor()(image)
        image_xyz = transforms.ToTensor()(image_xyz)
        mask = transforms.ToTensor()(mask)
        one_hot_label =  torch.tensor(one_hot_label)
        if self.normalize_fn:
            image = self.normalize_fn(image)
            image_xyz = self.normalize_fn(image_xyz)
        input.update({"image": image,"image_xyz": image_xyz, "mask": mask, "clslabel": one_hot_label})
        
        # print("**filename_xyz**", input[filename_xyz])
        
        return input
