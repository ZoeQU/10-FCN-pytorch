# -*- coding:utf-8 -*-
import pandas as pd
import os
import torch as t
import numpy as np
import torchvision.transforms.functional as ff
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

Train_ROOT = './CamVid/train'
Train_LABEL = './CamVid/train_labels'
Val_ROOT = './CamVid/val'
Val_LABEL = './CamVid/val_labels'
Test_ROOT = './CamVid/test'
Test_LABEL = './CamVid/test_labels'

train_imgs = os.listdir(Train_ROOT)
train_imgs = [os.path.join(Train_ROOT, img) for img in train_imgs]
train_imgs.sort()

train_labels = os.listdir(Train_LABEL)
train_labels = [os.path.join(Train_LABEL, label) for label in train_labels]
train_labels.sort()

val_imgs = os.listdir(Val_ROOT)
val_imgs = [os.path.join(Val_ROOT, img) for img in val_imgs]
val_imgs.sort()

val_labels = os.listdir(Val_LABEL)
val_labels = [os.path.join(Val_LABEL, label) for label in val_labels]
val_labels.sort()

test_imgs = os.listdir(Test_ROOT)
test_imgs = [os.path.join(Test_ROOT, img) for img in test_imgs]
test_imgs.sort()

test_labels = os.listdir(Test_LABEL)
test_labels = [os.path.join(Test_LABEL, label) for label in test_labels]
test_labels.sort()

pd_label_color = pd.read_csv('./CamVid/class_dict.csv', sep=',')
name_value = pd_label_color['name'].values  # ndarray type
num_class = len(name_value)

colormap = []
for i in range(len(pd_label_color.index)):
    # 'iloc' means index by row
    tmp = pd_label_color.iloc[i]
    color = []
    color.append(tmp['r'])
    color.append(tmp['g'])
    color.append(tmp['b'])
    colormap.append(color)
cm = np.array(colormap).astype('uint8')


def center_crop(data, label, crop_size):
    data = ff.center_crop(data, crop_size)
    label = ff.center_crop(label, crop_size)

    return data, label


################################################################
cm2lbl = np.zeros(256 ** 3)
for i, cm in enumerate(colormap):
    # customize a Hash code for colormap, to retrieval faster
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

def image2label(img):
    data = np.array(img, dtype='int32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(cm2lbl[idx], dtype='int64')
################################################################




def img_transform(img, label, crop_size):
    img, label = center_crop(img, label, crop_size)
    label = np.array(label)
    label = Image.fromarray(label.astype('uint8'))

    transform_img = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    ) # normalize fixed configs

    img = transform_img(img)
    label = image2label(label)
    label = t.from_numpy(label)

    return img, label


class CamvidDataset(Dataset):
    def __init__(self, mode=None, crop_size=None, transform=None):
        self.mode = mode

        self.train_imgs = train_imgs
        self.train_labels = train_labels

        self.val_imgs = val_imgs
        self.val_labels = val_labels

        self.test_imgs = test_imgs
        self.test_labels = test_labels

        if self.mode == 'train':
            self.imgs = self.train_imgs
            self.labels = self.train_labels

        elif self.mode == 'val':
            self.imgs = self.val_imgs
            self.labels = self.val_labels

        elif self.mode == 'test':
            self.imgs = self.test_imgs
            self.labels = self.test_labels

        self.crop_size = crop_size
        self.transforms = transform

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        img = Image.open(img)
        label = Image.open(label).convert('RGB')

        img, label = self.transforms(img, label, self.crop_size)

        sample = {'img': img, 'label': label}

        return sample

    def __len__(self):
        return len(self.imgs)


input_size = (352, 480)
Cam_train = CamvidDataset('train', input_size, img_transform)
# num_items = Cam_train.__len__()
# print(num_items)
# print(Cam_train.__getitem__(4))
Cam_val = CamvidDataset('val', input_size, img_transform)
# print(Cam_val.__len__())
Cam_test = CamvidDataset('test', input_size, img_transform)
# print(Cam_test.__len__())
# print("-------finish load data----------")