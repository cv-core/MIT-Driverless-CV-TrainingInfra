import concurrent.futures
import csv
import copy
import json
import math
import operator
import os
import sys
import random
import warnings
import numpy as np

import torchvision
import torch
import PIL
from PIL import Image, ImageDraw
import torch.utils.data
import torch.nn.functional as F
import imgaug.augmenters as iaa

from utils.utils import xyxy2xywh, xywh2xyxy, calculate_padding, visualize_and_save_to_local, scale_image, add_class_dimension_to_labels, xyhw2xyxy_corner, scale_labels, add_padding_on_each_side, get_patch_spacings, get_patch, pre_tile_padding, filter_and_offset_labels, upload_label_and_image_to_gcloud

##### section for all random seeds #####
torch.manual_seed(17)
np.random.seed(17)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
########################################

random.seed(a=17, version=2)
torchvision.set_image_backend('accimage')
visualization_tmp_path = "/outputs/visualization/"

class ImageLabelDataset(torch.utils.data.Dataset, object):
    def __init__(self, path, dataset_path, width, height, augment_affine, num_images, augment_hsv, lr_flip, ud_flip, bw, n_cpu, vis_batch, data_aug, blur, salt, noise, contrast, sharpen, ts,debug_mode, upload_dataset):
        self.img_files = []
        self.labels = []
        if ts:
            self.scales = []
        self.num_targets_per_image = None

        list_path = path

        self.ts = ts
        self.debug_mode = debug_mode

        with open(list_path) as csv_file:
            csv_reader = csv.reader(csv_file)
            for i, row in enumerate(csv_reader):
                if i < 2:
                    continue
                img_boxes = []
                for img_box_str in row[5:]:
                    if not img_box_str == "":
                        img_boxes.append(json.loads(img_box_str))

                img_boxes = torch.tensor(img_boxes, dtype=torch.float)
                if (img_boxes < 0).sum() > 0:
                    warnings.warn("Image {image} at line {line} has negative bounding box coordinates; skipping".format(image=os.path.join(dataset_path,row[0]), line=i+1))
                    continue

                img_width, img_height = int(row[2]), int(row[3])

                scale = float(row[4])

                new_height = int(img_height * scale)
                new_width = int(img_width * scale)

                vert_pad, horiz_pad = pre_tile_padding(new_width,new_height,width,height)

                if self.ts:
                    _,_,n_patches,_,_ = get_patch_spacings(new_width+horiz_pad*2, new_height+vert_pad*2, width, height)
                    self.img_files.extend([os.path.join(dataset_path,row[0])]*n_patches)
                    self.labels.extend([img_boxes]*n_patches)
                else:
                    self.img_files.append(os.path.join(dataset_path,row[0]))
                    self.labels.append(img_boxes)
                if ts:
                    self.scales.extend([float(row[4])]*n_patches)

        if num_images >= 0:
            sample_indices = random.sample(range(len(self.img_files)), k=num_images)
            if len(sample_indices) > 1:
                self.img_files = operator.itemgetter(*sample_indices)(self.img_files)
                self.labels = operator.itemgetter(*sample_indices)(self.labels)
                if ts:
                    self.scales = operator.itemgetter(*sample_indices)(self.scales)

        if n_cpu > 0:
            executor = concurrent.futures.ProcessPoolExecutor(n_cpu)
            futures = []

        for (img_file, label) in zip(self.img_files, self.labels):
            if self.num_targets_per_image is None or len(label) > self.num_targets_per_image:
                self.num_targets_per_image = len(label)

        if n_cpu > 0:
            concurrent.futures.wait(futures)

        self.height = height
        self.width = width
        self.augment_affine = augment_affine
        self.lr_flip = lr_flip
        self.ud_flip = ud_flip
        self.augment_hsv = augment_hsv
        self.data_aug = data_aug
        if self.augment_hsv or self.data_aug:
            self.jitter = torchvision.transforms.ColorJitter(saturation=0.25, contrast=0.25, brightness=0.25, hue=0.04)
        self.bw = bw # black and white
        self.vis_batch = vis_batch
        self.vis_counter = 0
        self.blur = blur
        self.salt = salt
        self.noise = noise
        self.contrast = contrast
        self.sharpen = sharpen
        self.upload_dataset = upload_dataset
        

    def __len__(self):
        return len(self.img_files)


    def __getitem__(self, index):
        img_uri = self.img_files[index]
        img_labels = self.labels[index]
        # don't download, since it was already downloaded in the init
        img_path = img_uri
        img_name = ("_".join(map(str, img_path.split("_")[-5:])))
        orig_img = PIL.Image.open(img_path).convert('RGB')
        if orig_img is None:
            raise Exception("Empty image: {img_path}".format(img_path=img_path))

        if self.vis_batch and len(img_labels) > 0:
            vis_orig_img = copy.deepcopy(orig_img)
            labels = add_class_dimension_to_labels(img_labels)
            labels = xyhw2xyxy_corner(labels, skip_class_dimension=True)
            tmp_path = os.path.join(visualization_tmp_path, img_name[:-4] + ".jpg")
            visualize_and_save_to_local(vis_orig_img, labels, tmp_path, box_color="green")
            print(f'new image uploaded to {tmp_path}')
        
        # First, handle image re-shaping 
        if self.ts:
            scale = self.scales[index]
            scaled_img = scale_image(orig_img, scale)
            scaled_img_width, scaled_img_height = scaled_img.size
            patch_width, patch_height = self.width, self.height

            vert_pad, horiz_pad = pre_tile_padding(scaled_img_width,scaled_img_height,patch_width,patch_height)
            padded_img = torchvision.transforms.functional.pad(scaled_img, padding=(horiz_pad, vert_pad, horiz_pad, vert_pad), fill=(127, 127, 127), padding_mode="constant")
            padded_img_width, padded_img_height = padded_img.size

            _,_,n_patches,_,_ = get_patch_spacings(padded_img_width, padded_img_height,
                                                   patch_width, patch_height)

            patch_index = random.randint(0,n_patches-1)
            if self.debug_mode:
                patch_index = 0
            img, boundary = get_patch(padded_img, patch_width, patch_height, patch_index)
        else:
            orig_img_width, orig_img_height = orig_img.size
            vert_pad, horiz_pad, ratio = calculate_padding(orig_img_height, orig_img_width, self.height, self.width)
            img = torchvision.transforms.functional.pad(orig_img, padding=(horiz_pad, vert_pad, horiz_pad, vert_pad), fill=(127, 127, 127), padding_mode="constant")
            img = torchvision.transforms.functional.resize(img, (self.height, self.width))

        # If no labels, no need to do augmentation (this should change in the future)
        #   so immediately return with the padded image and empty labels
        if len(img_labels) == 0:
            labels = torch.zeros((len(img_labels), 5))
            img = torchvision.transforms.functional.to_tensor(img)
            labels = F.pad(labels,
                    pad=[0, 0, 0, self.num_targets_per_image - len(labels)],
                    mode="constant")
            return img_uri, img, labels

        # Next, handle label re-shaping 
        labels = add_class_dimension_to_labels(img_labels)
        labels = xyhw2xyxy_corner(labels)
        if self.ts:
            labels = scale_labels(labels, self.scales[index])
            labels = add_padding_on_each_side(labels, horiz_pad, vert_pad)
            if self.vis_batch:
                tmp_path = os.path.join(visualization_tmp_path, img_name[:-4] + "_scaled.jpg")
                visualize_and_save_to_local(padded_img, labels, tmp_path, box_color="red")

            labels_temp = filter_and_offset_labels(labels, boundary)

            if self.vis_batch:
                pre_vis_labels = copy.deepcopy(labels)
                for i in range(n_patches):
                    vis_patch_img, boundary = get_patch(padded_img, patch_width, patch_height, i)

                    labels = filter_and_offset_labels(pre_vis_labels, boundary)

                    tmp_path = os.path.join(visualization_tmp_path, img_name[:-4] + \
                                        "_patch_{}.jpg".format(i))
                    visualize_and_save_to_local(vis_patch_img, labels, tmp_path, box_color="blue")
            if self.upload_dataset:
                pre_vis_labels = copy.deepcopy(labels)
                for i in range(n_patches):
                    vis_patch_img, boundary = get_patch(padded_img, patch_width, patch_height, i)

                    labels = filter_and_offset_labels(pre_vis_labels, boundary)

                    tmp_path = os.path.join(visualization_tmp_path, img_name[:-4] + \
                                        "_patch_{}.jpg".format(i))
                    upload_label_and_image_to_gcloud(vis_patch_img, labels, tmp_path)

            else:
                labels = filter_and_offset_labels(labels, boundary)
        else:
            labels = add_padding_on_each_side(labels, horiz_pad, vert_pad)
            labels = scale_labels(labels, ratio)
            labels_temp = labels

            if self.vis_batch:
                tmp_path = os.path.join(visualization_tmp_path, img_name[:-4] + "_pad_resized.jpg")
                visualize_and_save_to_local(img, labels, tmp_path, box_color="blue")

        labels = labels_temp
        if self.vis_batch and self.data_aug:
            vis_aug_img = copy.deepcopy(img)
            tmp_path = os.path.join(visualization_tmp_path, img_name[:-4] + "_before_aug.jpg")
            visualize_and_save_to_local(vis_aug_img, labels, tmp_path, box_color="red")
        if self.augment_hsv or self.data_aug:
            if random.random() > 0.5:
                img = self.jitter(img)
                # no transformation on labels

        # Augment image and labels
        img_width, img_height = img.size
        if self.augment_affine or self.data_aug:
            if random.random() > 0:
                angle = random.uniform(-10, 10)
                translate = (random.uniform(-40, 40), random.uniform(-40, 40)) ## WORKS
                scale = random.uniform(0.9, 1.1)
                shear = random.uniform(-3, 3)
                img = torchvision.transforms.functional.affine(img, angle, translate, scale, shear, 2, fillcolor=(127, 127, 127))
                labels = affine_labels(img_height, img_width, labels, -angle, translate, scale, (-shear, 0))

        if self.bw:
            img = torchvision.transforms.functional.to_grayscale(img, num_output_channels=1)
        
        # random left-right flip
        if self.lr_flip:
            if random.random() > 0.5:
                img = torchvision.transforms.functional.hflip(img)
                # Is this correct?
                # Not immediately obvious, when composed with the angle shift above
                labels[:, 1] = img_width - labels[:, 1]
                labels[:, 3] = img_width - labels[:, 3]

        # GaussianBlur, needs further development
        if self.blur:
            if random.random() > 0.2:
                arr = np.asarray(img)
                angle = random.uniform(40, -40)
                sigma = random.uniform(0,3.00)
                seq = iaa.Sequential([
                    iaa.GaussianBlur(sigma=sigma)
                    ])
                images_aug = seq.augment_images(arr)
                img = PIL.Image.fromarray(np.uint8(images_aug),'RGB')
        
        #AdditiveGaussianNoise
        if self.noise:
            if random.random() > 0.3:
                arr = np.asarray(img)
                scale = random.uniform(0,0.03*255)
                seq = iaa.Sequential([
                    iaa.AdditiveGaussianNoise(loc=0, scale=scale, per_channel=0.5)
                    ])
                images_aug = seq.augment_images(arr)
                img = PIL.Image.fromarray(np.uint8(images_aug),'RGB')

        #SigmoidContrast, need further development
        if self.contrast:
            if random.random() > 0.5:
                arr = np.asarray(img)
                cutoff = random.uniform(0.45,0.75)
                gain = random.randint(5,10)
                seq = iaa.Sequential([
                    iaa.SigmoidContrast(gain=gain,cutoff=cutoff) 
                    ])
                images_aug = seq.augment_images(arr)
                img = PIL.Image.fromarray(np.uint8(images_aug),'RGB')

        #Sharpen, need further development
        if self.sharpen:
            if random.random() > 0.3:
                arr = np.asarray(img)
                alpha = random.uniform(0,0.5)
                seq = iaa.Sharpen(alpha=alpha)
                images_aug = seq.augment_images(arr)
                img = PIL.Image.fromarray(np.uint8(images_aug),'RGB')

        if self.vis_batch and self.data_aug:
            vis_post_aug_img = copy.deepcopy(img)
            tmp_path = os.path.join(visualization_tmp_path, img_name[:-4] + "_post_augmentation.jpg")
            visualize_and_save_to_local(vis_post_aug_img, labels, tmp_path, box_color="green")

        if self.vis_batch:
            self.vis_counter += 1
            if self.vis_counter > (self.vis_batch -1):
                sys.exit('Finished visualizing enough images. Exiting!')

        labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
        labels[:, (1, 3)] /= self.width
        labels[:, (2, 4)] /= self.height

        img = torchvision.transforms.functional.to_tensor(img)
        labels = F.pad(labels, pad=[0, 0, 0, self.num_targets_per_image - len(labels)], mode="constant")
        if (labels < 0).sum() > 0:
            raise Exception(f"labels for image {img_uri} have negative values")
        return img_uri, img, labels

def affine_labels(h, w, targets, angle=0, translate=(0, 0), scale=1.0, shear=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    height = max(w, h)

    # Rotation and Scale
    alpha = scale * math.cos(math.radians(angle))
    beta = scale * math.sin(math.radians(angle))
    R = torch.tensor((
        (alpha, beta, (1-alpha)*(w/2.0)-beta*(h/2.0)),
        (-beta, alpha, (beta*w/2.0)+(1-alpha)*(h/2.0)),
        (0, 0, 1)
    ), dtype=torch.float)
    # angle += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations

    # Translation
    T = torch.eye(3)
    T[0, 2] = translate[0]  # x translation (pixels)
    T[1, 2] = translate[1]  # y translation (pixels)

    # Shear (about the center)
    S = torch.eye(3)
    S[0, 1] = math.tan(math.radians(shear[0])) # x shear
    S[0, 2] = -math.tan(math.radians(shear[0])) * h/2.0
    S[1, 0] = math.tan(math.radians(shear[1])) # y shear
    S[1, 2] = -math.tan(math.radians(shear[1])) * w/2.0
    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!

    # Return warped points also
    n = targets.shape[0]
    points = targets[:, 1:5]
    area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

    # warp points
    xy = torch.ones((n * 4, 3))
    xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
    xy = (xy @ M.transpose(0, 1))
    xy = xy[:, :2] / xy[:, 2].unsqueeze(1).expand(-1, 2)
    xy = xy[:, :2].reshape(n, 8)

    # create new boxes
    x = xy[:, [0, 2, 4, 6]]
    y = xy[:, [1, 3, 5, 7]]
    xy = torch.cat((x.min(1)[0], y.min(1)[0], x.max(1)[0], y.max(1)[0])).reshape(4, n).transpose(0, 1)

    # apply angle-based reduction
    radians = angle * math.pi / 180
    reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
    x = (xy[:, 2] + xy[:, 0]) / 2
    y = (xy[:, 3] + xy[:, 1]) / 2
    w = (xy[:, 2] - xy[:, 0]) * reduction
    h = (xy[:, 3] - xy[:, 1]) * reduction
    xy = torch.cat((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).transpose(0, 1)
    
    #print("diagnosing affine targets")
    # reject warped points outside of image
    torch.clamp(xy, 0, height, out=xy)
    w = xy[:, 2] - xy[:, 0]
    h = xy[:, 3] - xy[:, 1]
    area = w * h
    ar = torch.max(w / (h + 1e-16), h / (w + 1e-16))
    i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

    targets[i, 1:5] = xy[i]
    return targets
