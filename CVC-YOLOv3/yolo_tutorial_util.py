import os
import random
import tempfile
import time
import multiprocessing
import subprocess
import math
import shutil
import math

from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import Darknet
from utils.datasets import ImageLabelDataset
from utils.utils import model_info, print_args, Logger, visualize_and_save_to_local,xywh2xyxy
import validate
import warnings

import sys
from os.path import isfile, join
import copy
import cv2
from tensorboardX import SummaryWriter
from PIL import Image, ImageDraw
import torchvision
from utils.nms import nms
from utils.utils import calculate_padding
from tqdm import tqdm

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
num_cpu = multiprocessing.cpu_count() if cuda else 0

def run_epoch(label_prefix, data_loader, num_steps, optimizer, model, epoch, num_epochs, step, device):
    print(f"Model in {label_prefix} mode")
    epoch_losses = [0.0] * 7
    epoch_time_total = 0.0
    epoch_num_targets = 1e-12
    t1 = time.time()
    loss_labels = ["Total", "L-x", "L-y", "L-w", "L-h", "L-noobj", "L-obj"]
    for i, (img_uri, imgs, targets) in enumerate(data_loader):
        if step[0] >= num_steps:
            break
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        targets.requires_grad_(False)
        step_num_targets = ((targets[:, :, 1:5] > 0).sum(dim=2) > 1).sum().item() + 1e-12
        epoch_num_targets += step_num_targets
        # Compute loss, compute gradient, update parameters
        if optimizer is not None:
            optimizer.zero_grad()
        losses = model(imgs, targets)
        if label_prefix == "train":
            losses[0].sum().backward()
        if optimizer is not None:
            optimizer.step()

        for j, (label, loss) in enumerate(zip(loss_labels, losses)):
            batch_loss = loss.sum().to('cpu').item()
            epoch_losses[j] += batch_loss
        finished_time = time.time()
        step_time_total = finished_time - t1
        epoch_time_total += step_time_total
        
        statement = label_prefix + ' Epoch: ' + str(epoch) + ', Batch: ' + str(i + 1) + '/' + str(len(data_loader))
        count = 0
        for (loss_label, loss) in zip(loss_labels, losses):
            if count == 0:
                statement += ', Total: ' + '{0:10.6f}'.format(loss.item() / step_num_targets)
                tot_loss = loss.item()
                count += 1
            else:
                statement += ',   ' + loss_label + ': {0:5.2f}'.format(loss.item() / tot_loss * 100) + '%'
        print(statement)
        if label_prefix == "train":
            step[0] += 1
    return epoch_losses, epoch_time_total, epoch_num_targets


def single_img_detect(target_path,output_path,mode,model,device,conf_thres,nms_thres):

    img = Image.open(target_path).convert('RGB')
    w, h = img.size
    new_width, new_height = model.img_size()
    pad_h, pad_w, ratio = calculate_padding(h, w, new_height, new_width)
    img = torchvision.transforms.functional.pad(img, padding=(pad_w, pad_h, pad_w, pad_h), fill=(127, 127, 127), padding_mode="constant")
    img = torchvision.transforms.functional.resize(img, (new_height, new_width))

    bw = model.get_bw()
    if bw:
        img = torchvision.transforms.functional.to_grayscale(img, num_output_channels=1)

    img = torchvision.transforms.functional.to_tensor(img)
    img = img.unsqueeze(0)
    
    with torch.no_grad():
        model.eval()
        img = img.to(device, non_blocking=True)
        # output,first_layer,second_layer,third_layer = model(img)
        output = model(img)


        for detections in output:
            detections = detections[detections[:, 4] > conf_thres]
            box_corner = torch.zeros((detections.shape[0], 4), device=detections.device)
            xy = detections[:, 0:2]
            wh = detections[:, 2:4] / 2
            box_corner[:, 0:2] = xy - wh
            box_corner[:, 2:4] = xy + wh
            probabilities = detections[:, 4]
            nms_indices = nms(box_corner, probabilities, nms_thres)
            main_box_corner = box_corner[nms_indices]
            if nms_indices.shape[0] == 0:  
                continue
        img_with_boxes = Image.open(target_path)
        draw = ImageDraw.Draw(img_with_boxes)
        w, h = img_with_boxes.size

        for i in range(len(main_box_corner)):
            x0 = main_box_corner[i, 0].to('cpu').item() / ratio - pad_w
            y0 = main_box_corner[i, 1].to('cpu').item() / ratio - pad_h
            x1 = main_box_corner[i, 2].to('cpu').item() / ratio - pad_w
            y1 = main_box_corner[i, 3].to('cpu').item() / ratio - pad_h 
            draw.rectangle((x0, y0, x1, y1), outline="red")

        if mode == 'image':
            img_with_boxes.save(os.path.join(output_path,target_path.split('/')[-1]))
            return os.path.join(output_path,target_path.split('/')[-1])
        else:
            img_with_boxes.save(target_path)
            return target_path

def detect(target_path,
           output_path,
           model,
           device,
           conf_thres,
           nms_thres,
           detection_tmp_path):

        target_filepath = target_path

        img_formats = ['.jpg', '.jpeg', '.png', '.tif']
        vid_formats = ['.mov', '.avi', '.mp4']

        mode = None

        if os.path.splitext(target_filepath)[-1].lower() in img_formats:
            mode = 'image'
        
        elif os.path.splitext(target_filepath)[-1].lower() in vid_formats:
            mode = 'video'
        
        print("Detection Mode is: " + mode)

        raw_file_name = target_filepath.split('/')[-1].split('.')[0].split('_')[-4:]
        raw_file_name = '_'.join(raw_file_name)
        
        if mode == 'image':
            detection_path = single_img_detect(target_path=target_filepath,output_path=output_path,mode=mode,model=model,device=device,conf_thres=conf_thres,nms_thres=nms_thres)

            print(f'Please check output image at {detection_path}')

        elif mode == 'video':
            if os.path.exists(detection_tmp_path):
                shutil.rmtree(detection_tmp_path)  # delete output folder
            os.makedirs(detection_tmp_path)  # make new output folder

            vidcap = cv2.VideoCapture(target_filepath)
            success,image = vidcap.read()
            count = 0

            

            while success:
                cv2.imwrite(detection_tmp_path + "/frame%d.jpg" % count, image)     # save frame as JPEG file      
                success,image = vidcap.read()
                count += 1

            # Find OpenCV version
            (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

            if int(major_ver)  < 3 :
                fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
                print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
            else :
                fps = vidcap.get(cv2.CAP_PROP_FPS)
                print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
            vidcap.release(); 

            frame_array = []
            files = [f for f in os.listdir(detection_tmp_path) if isfile(join(detection_tmp_path, f))]
        
            #for sorting the file names properly
            files.sort(key = lambda x: int(x[5:-4]))
            for i in tqdm(files,desc='Doing Single Image Detection'):
                filename=detection_tmp_path + i
                
                detection_path = single_img_detect(target_path=filename,output_path=output_path,mode=mode,model=model,device=device,conf_thres=conf_thres,nms_thres=nms_thres)
                #reading each files
                img = cv2.imread(detection_path)
                height, width, layers = img.shape
                size = (width,height)
                frame_array.append(img)

            local_output_uri = output_path + raw_file_name + ".mp4"
            
            video_output = cv2.VideoWriter(local_output_uri,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

            for frame in tqdm(frame_array,desc='Creating Video'):
                # writing to a image array
                video_output.write(frame)
            video_output.release()
            shutil.rmtree(detection_tmp_path)