#!/usr/bin/python3

import argparse
import os
from os.path import isfile, join
import random
import tempfile
import time
import copy
import multiprocessing
import subprocess
import shutil
import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from PIL import Image, ImageDraw

import torchvision
from models import Darknet
from utils import storage_client
from utils.datasets import ImageLabelDataset
from utils.nms import nms
from utils.utils import xywh2xyxy, calculate_padding

import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

detection_tmp_path = "/tmp/detect/"

def download_file(file_uri):
    os_filepath = storage_client.get_file(file_uri)
    if not os.path.isfile(os_filepath):
        raise Exception("could not download file: {file_uri}".format(file_uri=file_uri))
    return os_filepath

def main(target_uri,
         output_uri,
         weights_uri,
         model_cfg,
         conf_thres,
         nms_thres,
         layer_sense,
         xy_loss,
         wh_loss,
         background_loss,
         foreground_loss,
         vanilla_anchor):

    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    random.seed(0)
    torch.manual_seed(0)
    if cuda:
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    model = Darknet(config_path=model_cfg,xy_loss=xy_loss,wh_loss=wh_loss,background_loss=background_loss,foreground_loss=foreground_loss,vanilla_anchor=vanilla_anchor)

    # Load weights
    weights_path = storage_client.get_file(weights_uri)
    model.load_weights(weights_path, model.get_start_weight_dim())
    model.to(device, non_blocking=True)

    detect(target_uri,
           output_uri,
           model,
           device=device,
           conf_thres=conf_thres,
           nms_thres=nms_thres,
           layer_sense=layer_sense)

def single_img_detect(target_uri,model,device,conf_thres,nms_thres,layer_sense):

    img = Image.open(target_uri).convert('RGB')
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
        if not layer_sense:
            img_with_boxes = Image.open(target_uri)
            draw = ImageDraw.Draw(img_with_boxes)
            w, h = img_with_boxes.size

            for i in range(len(main_box_corner)):
                x0 = main_box_corner[i, 0].to('cpu').item() / ratio - pad_w
                y0 = main_box_corner[i, 1].to('cpu').item() / ratio - pad_h
                x1 = main_box_corner[i, 2].to('cpu').item() / ratio - pad_w
                y1 = main_box_corner[i, 3].to('cpu').item() / ratio - pad_h 
                draw.rectangle((x0, y0, x1, y1), outline="red")

            img_with_boxes.save(target_uri)
            return target_uri
        else:
            ##### detections on first layer #####
            for detections in first_layer:
                detections = detections[detections[:, 4] > conf_thres]
                box_corner = torch.zeros((detections.shape[0], 4), device=detections.device)
                xy = detections[:, 0:2]
                wh = detections[:, 2:4] / 2
                box_corner[:, 0:2] = xy - wh
                box_corner[:, 2:4] = xy + wh
                probabilities = detections[:, 4]
                nms_indices = nms(box_corner, probabilities, nms_thres)
                first_box_corner = box_corner[nms_indices]
                if nms_indices.shape[0] == 0:  
                    continue
                
                true_first_box_corner = []
                for i in first_box_corner:
                    if i in main_box_corner:
                        true_first_box_corner.append(i)
                if not true_first_box_corner == []:
                    true_first_box_corner = torch.stack(true_first_box_corner)

                img_with_boxes = Image.open(target_uri)
                draw = ImageDraw.Draw(img_with_boxes)
                w, h = img_with_boxes.size

                for i in range(len(true_first_box_corner)):
                    x0 = true_first_box_corner[i, 0].to('cpu').item() / ratio - pad_w
                    y0 = true_first_box_corner[i, 1].to('cpu').item() / ratio - pad_h
                    x1 = true_first_box_corner[i, 2].to('cpu').item() / ratio - pad_w
                    y1 = true_first_box_corner[i, 3].to('cpu').item() / ratio - pad_h 
                    draw.rectangle((x0, y0, x1, y1), outline="red")
            ##### detections on second layer #####
            for detections in second_layer:
                detections = detections[detections[:, 4] > conf_thres]
                box_corner = torch.zeros((detections.shape[0], 4), device=detections.device)
                xy = detections[:, 0:2]
                wh = detections[:, 2:4] / 2
                box_corner[:, 0:2] = xy - wh
                box_corner[:, 2:4] = xy + wh
                probabilities = detections[:, 4]
                nms_indices = nms(box_corner, probabilities, nms_thres)
                second_box_corner = box_corner[nms_indices]
                if nms_indices.shape[0] == 0:  
                    continue
                
                true_second_box_corner = []
                for i in second_box_corner:
                    if i in main_box_corner:
                        true_second_box_corner.append(i)
                if not true_second_box_corner == []:
                    true_second_box_corner = torch.stack(true_second_box_corner)

                draw = ImageDraw.Draw(img_with_boxes)
                w, h = img_with_boxes.size

                for i in range(len(true_second_box_corner)):
                    x0 = true_second_box_corner[i, 0].to('cpu').item() / ratio - pad_w
                    y0 = true_second_box_corner[i, 1].to('cpu').item() / ratio - pad_h
                    x1 = true_second_box_corner[i, 2].to('cpu').item() / ratio - pad_w
                    y1 = true_second_box_corner[i, 3].to('cpu').item() / ratio - pad_h 
                    draw.rectangle((x0, y0, x1, y1), outline="black")
            ##### detections on third layer #####
            for detections in third_layer:
                detections = detections[detections[:, 4] > conf_thres]
                box_corner = torch.zeros((detections.shape[0], 4), device=detections.device)
                xy = detections[:, 0:2]
                wh = detections[:, 2:4] / 2
                box_corner[:, 0:2] = xy - wh
                box_corner[:, 2:4] = xy + wh
                probabilities = detections[:, 4]
                nms_indices = nms(box_corner, probabilities, nms_thres)
                third_box_corner = box_corner[nms_indices]
                if nms_indices.shape[0] == 0:  
                    continue
                
                true_third_box_corner = []
                for i in third_box_corner:
                    if i in main_box_corner:
                        true_third_box_corner.append(i)
                if not true_third_box_corner == []:
                    true_third_box_corner = torch.stack(true_third_box_corner)

                draw = ImageDraw.Draw(img_with_boxes)
                w, h = img_with_boxes.size

                for i in range(len(true_third_box_corner)):
                    x0 = true_third_box_corner[i, 0].to('cpu').item() / ratio - pad_w
                    y0 = true_third_box_corner[i, 1].to('cpu').item() / ratio - pad_h
                    x1 = true_third_box_corner[i, 2].to('cpu').item() / ratio - pad_w
                    y1 = true_third_box_corner[i, 3].to('cpu').item() / ratio - pad_h 
                    draw.rectangle((x0, y0, x1, y1), outline="blue")

                

            img_with_boxes.save(target_uri)
            return target_uri

def detect(target_uri,
           output_uri,
           model,
           device,
           conf_thres,
           nms_thres,
           layer_sense):

        if target_uri.startswith('gs'):
            target_filepath = download_file(target_uri)
        else:
            target_filepath = target_uri

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
            detection_path = single_img_detect(target_uri=target_filepath,model=model,device=device,conf_thres=conf_thres,nms_thres=nms_thres,layer_sense=layer_sense)

            gcloud_output_uri= output_uri + raw_file_name + ".jpg"

            storage_client.upload_file(detection_path, gcloud_output_uri)
            os.remove(detection_path)
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
                
                detection_path = single_img_detect(target_uri=filename,model=model,device=device,conf_thres=conf_thres,nms_thres=nms_thres,layer_sense=layer_sense)
                #reading each files
                img = cv2.imread(detection_path)
                height, width, layers = img.shape
                size = (width,height)
                frame_array.append(img)

            local_output_uri = detection_tmp_path + raw_file_name + ".mp4"
            gcloud_output_uri= output_uri + raw_file_name + ".mp4"
            
            video_output = cv2.VideoWriter(local_output_uri,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

            for frame in tqdm(frame_array,desc='Creating Video'):
                # writing to a image array
                video_output.write(frame)
            video_output.release()
            storage_client.upload_file(local_output_uri, gcloud_output_uri)
            shutil.rmtree(detection_tmp_path)
        print("Please go to the link below to check the detection output file: ")
        print(output_uri)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    def add_bool_arg(name, default, help):
        arg_group = parser.add_mutually_exclusive_group(required=False)
        arg_group.add_argument('--' + name, dest=name, action='store_true', help=help)
        arg_group.add_argument('--no_' + name, dest=name, action='store_false', help=("Do not " + help))
        parser.set_defaults(**{name:default})
    parser.add_argument('--model_cfg', type=str, default='model_cfg/yolo_baseline.cfg')
    parser.add_argument('--target_uri', type=str, default="gs://mit-dut-driverless-internal/data/videos/Round3/2019-04-26-19-17-44_pointgrey.mp4")
    parser.add_argument('--output_uri', type=str, default="gs://mit-dut-driverless-internal/dumping-ground/detection/")
    parser.add_argument('--weights_uri', type=str, default='gs://mit-dut-driverless-internal/vectorized-yolov3-training/july-2019-experiments/800800_k-means_anchors/16.weights')
    parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.25, help='IoU threshold for non-maximum suppression')

    add_bool_arg('layer_sense', default=False, help='whether to draw detection on each layer with different color')

    add_bool_arg('vanilla_anchor', default=False, help="whether to use vanilla anchor boxes for training")
    ##### Loss Constants #####
    parser.add_argument('--xy_loss', type=float, default=2, help='confidence loss for x and y')
    parser.add_argument('--wh_loss', type=float, default=1.6, help='confidence loss for width and height')
    parser.add_argument('--background_loss', type=float, default=25, help='confidence loss for background')
    parser.add_argument('--foreground_loss', type=float, default=0.1, help='confidence loss for foreground')

    opt = parser.parse_args()

    main(target_uri=opt.target_uri,
         output_uri=opt.output_uri,
         weights_uri=opt.weights_uri,
         model_cfg=opt.model_cfg,
         conf_thres=opt.conf_thres,
         nms_thres=opt.nms_thres,
         layer_sense=opt.layer_sense,
         xy_loss=opt.xy_loss,
         wh_loss=opt.wh_loss,
         background_loss=opt.background_loss,
         foreground_loss=opt.foreground_loss,
         vanilla_anchor=opt.vanilla_anchor)