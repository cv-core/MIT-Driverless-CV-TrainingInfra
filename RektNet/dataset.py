import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2
import os
import shutil
import PIL
from PIL import Image, ImageDraw
import random
import math
import sys
import copy
from utils import vis_kpt_and_save, vis_hm_and_save, prep_image, prep_label, get_scale, scale_labels

def print_tensor_stats(x, name):
    flattened_x = x.cpu().detach().numpy().flatten()
    avg = sum(flattened_x)/len(flattened_x)
    print(f"\t\t{name}: {avg},{min(flattened_x)},{max(flattened_x)}")

class ConeDataset(Dataset):
    def __init__(self, images, labels, dataset_path, target_image_size, save_checkpoints, vis_dataloader, transform=None):
        self.images = images
        self.labels = labels
        self.target_image_size = target_image_size
        self.transform = transform
        self.save_checkpoints = save_checkpoints
        self.vis_dataloader = vis_dataloader
        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = cv2.imread(self.dataset_path+self.images[index])
        orig_image_size = image.shape
        image_name = self.images[index].split(".")[0]
        image = prep_image(image=image,target_image_size=self.target_image_size)

        hm = prep_label(label=self.labels[index], target_image_size=self.target_image_size, orig_image_size=orig_image_size, image_path=self.images[index])
        h_scale, w_scale = get_scale(actual_image_size=orig_image_size, target_image_size=self.target_image_size)
        scaled_labels = scale_labels(self.labels[index], h_scale, w_scale)
        scaled_labels = scaled_labels / self.target_image_size[0]

        if self.vis_dataloader:
            tmp_image = copy.deepcopy(image)
            ##### visualize label #####

            vis_kpt_and_save(np_image=tmp_image, image_name=image_name, h_scale=h_scale, w_scale=w_scale, labels=scaled_labels)

            ##### visualize heat-map #####
            vis_hm_and_save(np_heat_map=hm, image_name=image_name)

        image = image.transpose((2, 0, 1)) / 255.0
        tensor_image = torch.from_numpy(image).type('torch.FloatTensor')
        return tensor_image, torch.from_numpy(hm).type('torch.FloatTensor'), torch.from_numpy(scaled_labels).type('torch.FloatTensor'), image_name, orig_image_size

