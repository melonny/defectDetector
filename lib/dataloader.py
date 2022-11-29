import os

import PIL
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import make_dataset, find_classes, IMG_EXTENSIONS, default_loader
from torchvision.transforms import transforms


def loadAEdata(opt):
    """ Load Data

    Args:
        opt ([type]): Argument Parser

    Raises:
        IOError: Cannot Load Dataset

    Returns:
        [type]: dataloader
    """
    if opt.dataroot == '':
        opt.dataroot = './data/{}'.format(opt.dataset)
    splits = ['train', 'test']
    drop_last_batch = {'train': True, 'test': False}
    shuffle = {'train': True, 'test': True}
    transform = transforms.Compose([transforms.Resize([opt.isize, opt.isize]),
                                    # transforms.CenterCrop(opt.isize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)), ])

    dataset = {x: ImageFolder(os.path.join(opt.dataroot, x), transform) for x in splits}
    dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                 batch_size=opt.batchsize,
                                                 shuffle=shuffle[x],
                                                 num_workers=int(opt.workers),
                                                 drop_last=drop_last_batch[x],
                                                 worker_init_fn=(None if opt.manualseed == -1
                                                                 else lambda x: np.random.seed(opt.manualseed)))
                  for x in splits}
    return dataloader

def loadSiamesedata(opt):
    if opt.dataroot == '':
        opt.dataroot = './data/{}'.format(opt.dataset)
    splits = ['train', 'test']
    # vgg11的输入尺寸: 224*224
    transform = transforms.Compose([transforms.Resize((opt.isize, opt.isize)),
                                    transforms.ToTensor()])
    # folder_dataset = torchvision.datasets.ImageFolder(root=training_dir)
    dataset = {x: ImageFolder(os.path.join(opt.dataroot, x), transform) for x in splits}
    if(opt.model == 'aesiamese'):
        siamese_dataset = {x:AEDataset(imageFolderDataset=dataset[x],transform=transform) for x in splits}
    else:
        siamese_dataset = {x:SiameseNetworkDataset(imageFolderDataset=dataset[x],
                                            transform=transform,
                                            should_invert=False
                                            ) for x in splits}
    # 定义图像dataloader
    train_dataloader = {x:DataLoader(dataset=siamese_dataset[x],
                                  shuffle=True,
                                  batch_size=opt.batchsize,
                                  num_workers=int(opt.workers)
                                  ) for x in splits}
    return train_dataloader

class AEDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self, index):
        img_tuple = self.imageFolderDataset.imgs[index]
        img = Image.open(img_tuple[0])
        img = img.convert("L")
        label = img_tuple[1]
        if self.transform is not None:
            img = self.transform(img)  # 在这里做transform，把图像转为tensor等等
        return img, label

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


class SiameseNetworkDataset(Dataset):
    """ SiameseNetworkDataset
    训练时同类样本约占一半

    return img0, img1, label

    label: 0 同类
    label: 1 异类
    """
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)  # 类别中任选一个
        # img0_tuple = 1
        should_get_same_class = random.randint(0, 1)  # 保证同类样本约占一半
        # should_get_same_class = 1
        if should_get_same_class:
            while True:
                # 直到找到同一类别
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # 直到找到非同一类别
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)