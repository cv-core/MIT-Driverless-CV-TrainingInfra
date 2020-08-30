import argparse
import tempfile
import sys
import os
import multiprocessing
import shutil
from tqdm import tqdm
import numpy as np
import cv2
import copy
from datetime import datetime
from tqdm import tqdm

import PIL
from PIL import Image, ImageDraw

import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

from keypoint_net import KeypointNet
from cross_ratio_loss import CrossRatioLoss
from utils import Logger
from utils import load_train_csv_dataset, prep_image, visualize_data, vis_tensor_and_save, calculate_distance, calculate_mean_distance
from dataset import ConeDataset

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

def print_tensor_stats(x, name):
    flattened_x = x.cpu().detach().numpy().flatten()
    avg = sum(flattened_x)/len(flattened_x)
    print(f"\t\t{name}: {avg},{min(flattened_x)},{max(flattened_x)}")

def eval_model(model, dataloader, loss_function, input_size):
    print("\tStarting validation...")
    model.eval()
    with torch.no_grad():
        loss_sums = [0,0,0]
        batch_num = 0
        for x_batch,y_hm_batch,y_point_batch,image_name, _ in dataloader:
            x_batch = x_batch.to(device)
            y_hm_batch = y_hm_batch.to(device)
            y_point_batch = y_point_batch.to(device)
            output = model(x_batch)
            loc_loss, geo_loss, loss = loss_function(output[0], output[1], y_hm_batch, y_point_batch)
            loss_sums[0] += loc_loss.item()
            loss_sums[1] += geo_loss.item()
            loss_sums[2] += loss.item()
            
            batch_num += 1

    val_loc_loss = loss_sums[0] / batch_num
    val_geo_loss = loss_sums[1] / batch_num
    val_loss = loss_sums[2] / batch_num
    print(f"\tValidation: MSE/Geometric/Total Loss: {round(val_loc_loss,10)}/{round(val_geo_loss,10)}/{round(val_loss,10)}")

    return val_loc_loss, val_geo_loss, val_loss

def print_kpt_L2_distance(model, dataloader, kpt_keys, study_name, evaluate_mode, input_size):
    kpt_distances = []
    if evaluate_mode:
        validation_textfile = open('logs/rektnet_validation.txt', 'a')

    for x_batch, y_hm_batch, y_point_batch, _, image_shape in dataloader:
        x_batch = x_batch.to(device)
        y_hm_batch = y_hm_batch.to(device)
        y_point_batch = y_point_batch.to(device)

        output = model(x_batch)

        pred_points = output[1]*x_batch.shape[1]
        pred_points = pred_points.data.cpu().numpy()
        pred_points *= input_size
        target_points = y_point_batch*x_batch.shape[1]
        target_points = target_points.data.cpu().numpy()
        target_points *= input_size

        kpt_dis = calculate_distance(target_points, pred_points)

        ##### for validation knowledge of avg kpt mse vs BB size distribution #####
        if evaluate_mode:
            height,width,_ = image_shape
            print(width.numpy()[0],height.numpy()[0])
            print(kpt_dis)

            single_img_kpt_dis_sum = sum(kpt_dis) 
            validation_textfile.write(f"{[width.numpy()[0],height.numpy()[0]]}:{single_img_kpt_dis_sum}\n")
        ###########################################################################

        kpt_distances.append(kpt_dis)
    if evaluate_mode:
        validation_textfile.close()
    final_stats, total_dist, final_stats_std = calculate_mean_distance(kpt_distances)
    print(f'Mean distance error of each keypoint is:')
    for i, kpt_key in enumerate(kpt_keys):
        print(f'\t{kpt_key}: {final_stats[i]}')
    print(f'Standard deviation of each keypoint is:')
    for i, kpt_key in enumerate(kpt_keys):
        print(f'\t{kpt_key}: {final_stats_std[i]}')
    print(f'Total distance error is: {total_dist}')
    ##### updating best result for optuna study #####
    result = open("logs/" + study_name + ".txt", "w" )
    result.write(str(total_dist))
    result.close() 
    ###########################################