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

cv2.setRNGSeed(17)
torch.manual_seed(17)
np.random.seed(17)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

visualization_tmp_path = "/outputs/visualization/"

def print_tensor_stats(x, name):
    flattened_x = x.cpu().detach().numpy().flatten()
    avg = sum(flattened_x)/len(flattened_x)
    print(f"\t\t{name}: {avg},{min(flattened_x)},{max(flattened_x)}")

def train_model(model, output_uri, dataloader, loss_function, optimizer, scheduler, epochs, val_dataloader, intervals, input_size, num_kpt, save_checkpoints, kpt_keys, study_name, evaluate_mode):
    
    best_val_loss = float('inf')
    best_epoch = 0
    max_tolerance = 8
    tolerance = 0

    for epoch in range(epochs):
        print(f"EPOCH {epoch}")
        model.train()
        total_loss = [0,0,0] # location/geometric/total
        batch_num = 0

        train_process = tqdm(dataloader)
        for x_batch, y_hm_batch, y_points_batch, image_name, _ in train_process:
            x_batch = x_batch.to(device)
            y_hm_batch = y_hm_batch.to(device)
            y_points_batch = y_points_batch.to(device)

            # Zero the gradients.
            if optimizer is not None:
                optimizer.zero_grad()

            # Compute output and loss.
            output = model(x_batch)
            loc_loss, geo_loss, loss = loss_function(output[0], output[1], y_hm_batch, y_points_batch)
            loss.backward()
            optimizer.step()

            loc_loss, geo_loss, loss = loc_loss.item(), geo_loss.item(), loss.item()
            train_process.set_description(f"Batch {batch_num}. Location Loss: {round(loc_loss,5)}. Geo Loss: {round(geo_loss,5)}. Total Loss: {round(loss,5)}")
            total_loss[0] += loc_loss
            total_loss[1] += geo_loss
            total_loss[2] += loss
            batch_num += 1

        print(f"\tTraining: MSE/Geometric/Total Loss: {round(total_loss[0]/batch_num,10)}/{round(total_loss[1]/batch_num,10)}/{round(total_loss[2]/batch_num,10)}")
        val_loc_loss, val_geo_loss, val_loss = eval_model(model=model, dataloader=val_dataloader, loss_function=loss_function, input_size=input_size)

        # Position suggested by https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            tolerance = 0

            # Save model onnx for inference.
            if save_checkpoints:
                onnx_uri = os.path.join(output_uri,f"best_keypoints_{input_size[0]}{input_size[1]}.onnx")
                onnx_model = KeypointNet(num_kpt, input_size, onnx_mode=True)
                onnx_model.load_state_dict(model.state_dict())
                torch.onnx.export(onnx_model, torch.randn(1, 3, input_size[0], input_size[1]), onnx_uri)
                print(f"Saving ONNX model to {onnx_uri}")
                best_model = copy.deepcopy(model)
        else:
            tolerance += 1

        if save_checkpoints and epoch != 0 and (epoch + 1) % intervals == 0:
            # Save the latest weights
            gs_pt_uri = os.path.join(output_uri, "{epoch}_loss_{loss}.pt".format(epoch=epoch, loss=round(val_loss, 2)))
            print(f'Saving model to {gs_pt_uri}')
            checkpoint = {'epoch': epoch,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, gs_pt_uri)
        if tolerance >= max_tolerance:
            print(f"Training is stopped due; loss no longer decreases. Epoch {best_epoch} is has the best validation loss.")
            break

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

def main():
    def add_bool_arg(name, default, help):
        arg_group = parser.add_mutually_exclusive_group(required=False)
        arg_group.add_argument('--' + name, dest=name, action='store_true', help=help)
        arg_group.add_argument('--no_' + name, dest=name, action='store_false', help=("Do not " + help))
        parser.set_defaults(**{name:default})

    parser = argparse.ArgumentParser(description='Keypoints Training with Pytorch')

    parser.add_argument('--input_size', default=80, help='input image size')
    parser.add_argument('--train_dataset_uri', default='dataset/rektnet_label.csv', help='training dataset csv directory path')
    parser.add_argument('--output_path', type=str, help='output weights path, by default we will create a folder based on current system time and name of your cfg file',default="automatic")
    parser.add_argument('--dataset_path', type=str, help='path to image dataset',default="dataset/RektNet_Dataset/")
    parser.add_argument('--loss_type', default='l1_softargmax', help='loss type: l2_softargmax|l2_heatmap|l1_softargmax')
    parser.add_argument('--validation_ratio', default=0.15, type=float, help='percent of dataset to use for validation')
    parser.add_argument('--batch_size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float, help='learning rate')
    parser.add_argument('--lr_gamma', default=0.999, help='gamma for the scheduler')
    parser.add_argument('--num_epochs', default=1024, type=int, help='number of epochs')
    parser.add_argument("--checkpoint_interval", type=int, default=4, help="interval between saving model weights")
    parser.add_argument('--study_name', required=True, help='name for saving checkpoint models')

    add_bool_arg('geo_loss', default=True, help='whether to add in geo loss')
    parser.add_argument('--geo_loss_gamma_vert', default=0, type=float, help='gamma for the geometric loss (horizontal)')
    parser.add_argument('--geo_loss_gamma_horz', default=0, type=float, help='gamma for the geometric loss (vertical)')
    
    add_bool_arg('vis_upload_data', default=False, help='whether to visualize our dataset in Christmas Tree format and upload the whole dataset to. default to False')
    add_bool_arg('save_checkpoints', default=True, help='whether to save checkpoints')
    add_bool_arg('vis_dataloader', default=False, help='whether to visualize the image points and heatmap processed in our dataloader')
    add_bool_arg('evaluate_mode', default=False, help='whether to evaluate avg kpt mse vs BB size distribution at end of training')

    args = parser.parse_args()
    print("Program arguments:", args)

    if args.output_path == "automatic":
        current_month = datetime.now().strftime('%B').lower()
        current_year = str(datetime.now().year)
        if not os.path.exists(os.path.join('outputs/', current_month + '-' + current_year + '-experiments/' + args.study_name + '/')):
            os.makedirs(os.path.join('outputs/', current_month + '-' + current_year + '-experiments/' + args.study_name + '/'))
        output_uri = os.path.join('outputs/', current_month + '-' + current_year + '-experiments/' + args.study_name + '/')
    else:
        output_uri = args.output_path
    
    save_file_name = 'logs/' + output_uri.split('/')[-2]
    sys.stdout = Logger(save_file_name + '.log')
    sys.stderr = Logger(save_file_name + '.error')
    
    INPUT_SIZE = (args.input_size, args.input_size)
    KPT_KEYS = ["top", "mid_L_top", "mid_R_top", "mid_L_bot", "mid_R_bot", "bot_L", "bot_R"]

    intervals = args.checkpoint_interval
    val_split = args.validation_ratio
    
    batch_size= args.batch_size
    num_epochs= args.num_epochs

    # Load the train data.
    train_csv = args.train_dataset_uri
    train_images, train_labels, val_images, val_labels = load_train_csv_dataset(train_csv, validation_percent=val_split, keypoint_keys=KPT_KEYS, dataset_path=args.dataset_path, cache_location="./gs/")

    # "Become one with the data" - Andrej Karpathy
    if args.vis_upload_data:
        visualize_data(train_images, train_labels)
        print('Shutting down instance...')
        os.system('sudo shutdown now')

    # Create pytorch dataloaders for train and validation sets.
    train_dataset = ConeDataset(images=train_images, labels=train_labels, dataset_path=args.dataset_path, target_image_size=INPUT_SIZE, save_checkpoints=args.save_checkpoints, vis_dataloader=args.vis_dataloader)
    train_dataloader = DataLoader(train_dataset, batch_size= batch_size, shuffle=False, num_workers=0)
    val_dataset = ConeDataset(images=val_images, labels=val_labels, dataset_path=args.dataset_path, target_image_size=INPUT_SIZE, save_checkpoints=args.save_checkpoints, vis_dataloader=args.vis_dataloader)
    val_dataloader = DataLoader(val_dataset, batch_size= 1, shuffle=False, num_workers=0)

    # Define model, optimizer and loss function.
    model = KeypointNet(len(KPT_KEYS), INPUT_SIZE, onnx_mode=False)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    loss_func = CrossRatioLoss(args.loss_type, args.geo_loss, args.geo_loss_gamma_horz, args.geo_loss_gamma_vert)

    # Train our model.
    train_model(
        model=model,
        output_uri=output_uri,
        dataloader=train_dataloader, 
        loss_function=loss_func, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        epochs=num_epochs, 
        val_dataloader=val_dataloader, 
        intervals=intervals, 
        input_size=INPUT_SIZE,
        num_kpt=len(KPT_KEYS), 
        save_checkpoints=args.save_checkpoints,
        kpt_keys=KPT_KEYS,
        study_name=args.study_name,
        evaluate_mode=args.evaluate_mode
    )

if __name__=='__main__':
    main()
