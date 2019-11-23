#!/usr/bin/python3

import argparse
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
import multiprocessing

from models import Darknet
import torchvision
import torch
import PIL
from PIL import Image, ImageDraw
import torch.utils.data
import torch.nn.functional as F
from google.cloud import storage
from tqdm import tqdm

from utils import storage_client
from utils.utils import  calculate_padding, visualize_and_save_to_gcloud, add_class_dimension_to_labels, xyhw2xyxy_corner, add_padding_on_each_side

warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
n_cpu = multiprocessing.cpu_count() if cuda else 0
if cuda:
    torch.cuda.synchronize()

gc_storage_client = storage.Client(project="mitdriverless")
direc = './gs'

gcloud_tmp_path = "/tmp/"
gcloud_vet_path = "gs://mit-dut-driverless-internal/dumping-ground/yolo_vet/"

def download_image(image_uri):
    os_filepath = storage_client.get_file(image_uri)
    if not os.path.isfile(os_filepath):
        raise Exception("could not download image: {image_uri}".format(image_uri=image_uri))

def main(csv_uri, width, height, num_images,output_path):
    img_files = []
    img_labels = []
    num_targets_per_image = None

    list_path = storage_client.get_file(csv_uri)

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
                warnings.warn("Image {image} at line {line} has negative bounding box coordinates; skipping".format(image=row[1], line=i+1))
                continue

            img_files.append(row[1])
            img_labels.append(img_boxes)

    if num_images >= 0:
        sample_indices = random.sample(range(len(img_files)), k=num_images)
        if len(sample_indices) > 1:
            img_files = operator.itemgetter(*sample_indices)(img_files)
            img_labels = operator.itemgetter(*sample_indices)(img_labels)
    
    if n_cpu > 0:
        executor = concurrent.futures.ProcessPoolExecutor(n_cpu)
        futures = []

    for (img_file, img_label) in zip(img_files, img_labels):
        if n_cpu > 0:
            futures.append(executor.submit(download_image, img_file))
        else:
            download_image(img_file)
        if num_targets_per_image is None or len(img_label) > num_targets_per_image:
            num_targets_per_image = len(img_label)

    if n_cpu > 0:
        concurrent.futures.wait(futures)

    vis_process = tqdm(img_files)
    for index,_ in tqdm(enumerate(vis_process),desc="Uploading Visualization"):
        img_uri = img_files[index]
        img_path = storage_client.get_uri_filepath(img_uri)
        image_label = img_labels[index]
        img_name = ("_".join(map(str, img_path.split("_")[-5:])))
        orig_img = PIL.Image.open(img_path).convert('RGB')
        if orig_img is None:
            raise Exception("Empty image: {img_path}".format(img_path=img_path))

        orig_img_width, orig_img_height = orig_img.size
        vert_pad, horiz_pad, ratio = calculate_padding(orig_img_height, orig_img_width, height, width)
        img = torchvision.transforms.functional.pad(orig_img, padding=(horiz_pad, vert_pad, horiz_pad, vert_pad), fill=(127, 127, 127), padding_mode="constant")
        img = torchvision.transforms.functional.resize(img, (height, width))

        if len(image_label) == 0 or image_label.size()[0]==5:
            continue
        
        labels = add_class_dimension_to_labels(image_label)
        labels = xyhw2xyxy_corner(labels)

        labels = add_padding_on_each_side(labels, horiz_pad, vert_pad)
        labels = scale_labels(labels, ratio)

        gcloud_path = gcloud_vet_path + img_name
        tmp_path = os.path.join(gcloud_tmp_path, img_name)
        visualize_and_save_to_gcloud(img, labels, gcloud_path, tmp_path, box_color="red")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Yolo Visualization')

    parser.add_argument('--model_cfg', type=str, help='cfg file path',required=True)
    parser.add_argument('--output_path', help='path to upload the dataset for vetting', default="gs://mit-dut-driverless-internal/dumping-ground/yolo_vet/")

    args = parser.parse_args(sys.argv[1:])

    print("Initializing model")
    model = Darknet(args.model_cfg)
    img_width, img_height = model.img_size()
    validate_uri, train_uri, _, _ = model.get_links()
    num_validate_images, num_train_images = model.num_images()

    ##### uploading training images with label on for vetting #####
    print(f"uploading training images")
    main(csv_uri=train_uri,
    width=img_width, 
    height=img_height, 
    num_images=num_train_images,
    output_path=args.output_path)

    ##### uploading validation images with label on for vetting #####
    print(f"uploading validation images")
    main(csv_uri=validate_uri,
    width=img_width, 
    height=img_height, 
    num_images=num_validate_images,
    output_path=args.output_path)

    ###############################################################
