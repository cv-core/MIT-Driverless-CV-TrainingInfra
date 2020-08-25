import math
import numpy as np
import pandas as pd
import hashlib
import sys
import os
import shutil
import cv2
import tempfile
from google.cloud import storage
from tqdm import tqdm
import statistics
import torch

import PIL
from PIL import Image, ImageDraw

vis_tmp_path = "/tmp/detect/" #!!!don't specify this path outside of /tmp/, otherwise important files could be removed!!!
vis_path = "/outputs/visualization/"

if os.path.exists(vis_tmp_path):
    shutil.rmtree(vis_tmp_path)  # delete output folder
os.makedirs(vis_tmp_path)  # make new output folder

class Logger(object):
    def __init__(self, File):
        Type = File.split('.')[-1] 
        if Type == 'error':
            self.terminal = sys.stderr
        elif Type == 'log':
            self.terminal = sys.stdout
        self.log = open(File, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass 

def vis_kpt_and_save(np_image, image_name, h_scale, w_scale, labels, color=(52,31,163)):
    circ_size = 3
    for pt in np.array(labels):
        x_coor, y_coor = pt 
        cv2.circle(np_image, (x_coor, y_coor), circ_size, color, -1) #BGR color (52,31,163) is called mit logo red
    if not cv2.imwrite(os.path.join(vis_tmp_path, image_name + "_label_vis.jpg"), np_image):
        raise Exception("Could not write image")    #opencv won't give you error for incorrect image but return False instead, so we have to do it manually
    os.rename(os.path.join(vis_tmp_path, image_name + "_label_vis.jpg"), os.path.join(vis_path, image_name + "_label_vis.jpg"))

def vis_hm_and_save(np_heat_map, image_name):
    np_image = np.zeros((1, np_heat_map.shape[1], np_heat_map.shape[2]))
    for i in range(np_heat_map.shape[0]):
        np_image += np_heat_map[i,:,:] #sum up the heat-map numpy matrix
    data = np_image.astype('f')
    data = data.squeeze(0) #squeeze the numpy image size from (1,width,height) to (width,height)
    img = Image.fromarray(((data - data.min()) * 255.0 /
        (data.max() - data.min())).astype(np.uint8)) #convert to PIL image
    img.save(os.path.join(vis_tmp_path, image_name + "_heat_map.jpg")) # opencv doesn't like our heat-map, so we use PIL instead here
    os.rename(os.path.join(vis_tmp_path, image_name + "_heat_map.jpg"), os.path.join(vis_path, image_name + "_heat_map.jpg"))

def vis_tensor_and_save(image, h, w, tensor_output, image_name, output_uri):
    colors = [(0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (127, 255, 127), (255, 127, 127)]
    i = 0
    for pt in np.array(tensor_output):
        cv2.circle(image, (int(pt[0] * w), int(pt[1] * h)), 2, colors[i], -1)
        i += 1
    if not cv2.imwrite(os.path.join(vis_tmp_path, image_name + "_inference.jpg"), image):
        raise Exception("Could not write image")    #opencv won't give you error for incorrect image but return False instead, so we have to do it manually
    
    os.rename(os.path.join(vis_tmp_path, image_name + "_inference.jpg"), os.path.join(output_uri, image_name + "_inference.jpg"))
    return image

def prep_image(image,target_image_size):
    h,w,_ = image.shape
    image = cv2.resize(image, target_image_size)
    return image

def print_tensor_stats(x, name):
    flattened_x = x.cpu().detach().numpy().flatten()
    avg = sum(flattened_x)/len(flattened_x)
    print(f"\t\t{name}: {avg},{min(flattened_x)},{max(flattened_x)}")

def prep_label(label, target_image_size, orig_image_size, image_path):
    hm = np.zeros((label.shape[0], target_image_size[0], target_image_size[1]))
    for i in range(label.shape[0]):
        row = label[i]
        # padded_image_size = max(orig_image_size[0],orig_image_size[1])
        hm_tmp = np.zeros((orig_image_size[0], orig_image_size[1]))
        hm_tmp[int(row[1]), int(row[0])] = 1.0
        hm[i] = cv2.resize(hm_tmp, target_image_size)
        hm[i] = cv2.GaussianBlur(hm[i], (5,5), 0)
        if hm[i].sum()==0:
            print("Incorrect Data Label Detected! Please revise the image label below and becoming the one with data!")
            print(image_path)
        hm[i] /= hm[i].sum()
    return hm

def get_scale(actual_image_size,target_image_size):
    ##### since we are dealing with square image only, is doesn't matter we use height or width #####
    target_h, target_w = target_image_size
    h_scale = target_h / actual_image_size[0]
    w_scale = target_w / actual_image_size[1]
    return h_scale, w_scale

def scale_labels(labels, h_scale, w_scale):
    new_labels = []
    for pt in np.array(labels):
        x_coor = math.ceil((int(pt[0])) * w_scale)
        y_coor = math.ceil((int(pt[1])) * h_scale)
        new_labels.append([x_coor, y_coor])
    return np.asarray(new_labels)

def visualize_data(images, labels):
    vis_process = tqdm(images)
    for index,_ in tqdm(enumerate(vis_process),desc="Processing Visualization"):
        # print("{}/{}: {}".format(index + 1, len(images), images[index]))
        image = cv2.imread("./gs/"+images[index])
        h, w, _ = image.shape
        dim_diff = np.abs(h - w)
        if h<=w:
            image = cv2.copyMakeBorder(image, 0, dim_diff, 0, 0, cv2.BORDER_CONSTANT, value=[128,128,128])
        else:
            image = cv2.copyMakeBorder(image, 0, 0, 0, dim_diff, cv2.BORDER_CONSTANT, value=[128,128,128])
        h, w, _ = image.shape
        image = cv2.resize(image, (1000, 1000))
        label = labels[index]
        hm = np.zeros((label.shape[0], 1000, 1000))
        for i in range(hm.shape[0]):
            row = label[i]
            hm_tmp = np.zeros((h, w))
            hm_tmp[int(row[1]), int(row[0])] = 1.0
            hm[i] = cv2.resize(hm_tmp, (1000, 1000))
            hm[i] = cv2.GaussianBlur(hm[i], (5,5), 0)
            hm[i] /= hm[i].sum()
        label = label / np.array([w, h])[np.newaxis, :]
        h, w, _ = image.shape
        prevpt = None
        for pt in label:
            cvpt = (int(pt[0] * w), int(pt[1] * h))
            cv2.circle(image, cvpt, 3, (0, 255, 0), -1)
            if prevpt is not None:
                cv2.line(image, prevpt, cvpt, (0, 255, 0), 2)
            prevpt = cvpt
        cv2.imwrite(vis_tmp_path + images[index], image)
        os.rename(vis_tmp_path + images[index], vis_path + images[index])
        cv2.waitKey(0)
        for i in range(hm.shape[0]):
            cv2.imwrite(vis_tmp_path + images[index], image)
            os.rename(vis_tmp_path + images[index], vis_path + images[index])
            cv2.waitKey(0)

def load_train_csv_dataset(train_csv_uri, validation_percent, keypoint_keys, dataset_path, cache_location=None):
    train_data_table = pd.read_csv(train_csv_uri)
    train_data_table_hash = hashlib.sha256(pd.util.hash_pandas_object(train_data_table, index=True).values).hexdigest()

    train_images, train_labels = None, None
    if cache_location:
        cache_folder = os.path.join(cache_location, train_data_table_hash)
        cache_images_path = os.path.join(cache_folder, 'images.npy')
        cache_labels_path = os.path.join(cache_folder, 'labels.npy')

        if os.path.exists(cache_images_path) and os.path.exists(cache_labels_path):
            print(f"Caches exist: {cache_images_path} and {cache_labels_path}!")
            train_images = np.load(cache_images_path)
            train_labels = np.load(cache_labels_path)
        else:
            print("Caches do not exist!")

    if train_labels is None:
        # Separate the labels from the input data.
        images = train_data_table.values[:, 0]
        labels = train_data_table.values[:, 2:2+len(keypoint_keys)]

        tmp_labels = []
        image_uris = []

        for i in range(len(labels)):
            label = labels[i]
            if label[0] != label[0]:
                continue
            label_np = np.zeros((len(keypoint_keys), 2))
            for j in range(len(keypoint_keys)):
                col = keypoint_keys[j]
                txt = label[train_data_table.columns.get_loc(col) - 2][1:-1].split(",")
                label_np[j, 0] = txt[0]
                label_np[j, 1] = txt[1]
            tmp_labels.append(label_np)
            image_uris.append(os.path.join(dataset_path,images[i]))

        train_images = []
        train_labels = []

        # if not os.path.isdir("./gs"):
        #     os.mkdir("./gs")
        # print("Downloading dataset...")

        num = 0
        for uri in tqdm(image_uris,desc="Processing Image Dataset"):
            uri_parts = uri.split("/")

            image = cv2.imread(uri)
            h, _, _ = image.shape
            if h < 10:
                num += 1
                continue
            train_images.append(uri_parts[-1])
            train_labels.append(tmp_labels[num])
            num += 1

        if cache_location:
            print("Saving cache...")
            cache_folder = os.path.join(cache_location, train_data_table_hash)
            os.makedirs(cache_folder, exist_ok=True)
            cache_images_path = os.path.join(cache_folder, 'images.npy')
            cache_labels_path = os.path.join(cache_folder, 'labels.npy')
            print(cache_images_path, cache_labels_path)
            np.save(cache_images_path, train_images)
            np.save(cache_labels_path, train_labels)

    # Calculate how much of our training data is for train and validation.
    num_train = len(train_labels)
    num_val = int(num_train * validation_percent)

    # # Reshape data back to images, transpose to N,C,H,W format for pytorch.
    # train_images = train_images.reshape([-1, 28, 28, 1]).transpose((0, 3, 1, 2))
    
    # # Split for train/val.
    val_labels = train_labels[0:num_val]
    val_images = train_images[0:num_val]
    train_labels = train_labels[num_val:]
    train_images = train_images[num_val:]
    print(f"training image number: {len(train_images)}")
    print(f"validation image number: {len(val_images)}")

    return train_images, train_labels, val_images, val_labels



def calculate_distance(target_points,pred_points):
    dist_matrix = []
    for i, point  in enumerate(target_points[0]):
        dist = np.sqrt(np.square(point[0] - pred_points[0][i][0]) + np.square(point[1] - pred_points[0][i][1]))
        dist_matrix.append(dist)
    return dist_matrix

def calculate_mean_distance(epoch_kpt_dis):
    top = []
    mid_R_top = []
    mid_R_bot = []
    bot_R = []
    bot_L = []
    mid_L_bot = []
    mid_L_top = []
    for i, dist in enumerate(epoch_kpt_dis):
        top.append(dist[0])
        mid_L_top.append(dist[1])
        mid_R_top.append(dist[2])
        mid_L_bot.append(dist[3])
        mid_R_bot.append(dist[4])
        bot_L.append(dist[5])
        bot_R.append(dist[6])
        
    top_std = np.std(top)
    top = np.mean(top)

    mid_L_top_std = np.std(mid_L_top)
    mid_L_top = np.mean(mid_L_top)

    mid_R_top_std = np.std(mid_R_top)
    mid_R_top = np.mean(mid_R_top)
    
    mid_L_bot_std = np.std(mid_L_bot)
    mid_L_bot = np.mean(mid_L_bot)

    mid_R_bot_std = np.std(mid_R_bot)
    mid_R_bot = np.mean(mid_R_bot)
    
    bot_L_std = np.std(bot_L)
    bot_L = np.mean(bot_L)

    bot_R_std = np.std(bot_R)
    bot_R = np.mean(bot_R)


    total = top + mid_L_top + mid_R_top + mid_L_bot + mid_R_bot + bot_L + bot_R

    return [top,mid_L_top,mid_R_top,mid_L_bot,mid_R_bot,bot_L,bot_R],total,[top_std,mid_L_top_std,mid_R_top_std,mid_L_bot_std,mid_R_bot_std,bot_L_std,bot_R_std]

