#!/usr/bin/python3

import argparse
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

##### section for all random seeds #####
torch.manual_seed(17)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
########################################

warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
num_cpu = multiprocessing.cpu_count() if cuda else 0
if cuda:
    torch.cuda.synchronize()
random.seed(0)
torch.manual_seed(0)

if cuda:
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

def run_epoch(label_prefix, data_loader, num_steps, optimizer, model, epoch,
              num_epochs, step):
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

def main(*, evaluate, batch_size, optimizer_pick, model_cfg, weights_path, output_path, dataset_path, num_epochs, num_steps, checkpoint_interval, 
        augment_affine, augment_hsv, lr_flip, ud_flip, momentum, gamma, lr, weight_decay, vis_batch, data_aug, blur, salt, noise, contrast, sharpen, ts, debug_mode, upload_dataset,xy_loss,wh_loss,no_object_loss,object_loss,vanilla_anchor,val_tolerance,min_epochs):
    input_arguments = list(locals().items())

    print("Initializing model")
    model = Darknet(config_path=model_cfg,xy_loss=xy_loss,wh_loss=wh_loss,no_object_loss=no_object_loss,object_loss=object_loss,vanilla_anchor=vanilla_anchor)
    img_width, img_height = model.img_size()
    bw  = model.get_bw()
    validate_uri, train_uri = model.get_links()

    if output_path == "automatic":
        current_month = datetime.now().strftime('%B').lower()
        current_year = str(datetime.now().year)
        if not os.path.exists(os.path.join('outputs/', current_month + '-' + current_year + '-experiments/' + model_cfg.split('.')[0].split('/')[-1])):
            os.makedirs(os.path.join('outputs/', current_month + '-' + current_year + '-experiments/' + model_cfg.split('.')[0].split('/')[-1]))
        output_uri = os.path.join('outputs/', current_month + '-' + current_year + '-experiments/' + model_cfg.split('.')[0].split('/')[-1])
    else:
        output_uri = output_path

    num_validate_images, num_train_images = model.num_images()
    conf_thresh, nms_thresh, iou_thresh = model.get_threshs()
    num_classes = model.get_num_classes()
    loss_constant = model.get_loss_constant()
    conv_activation = model.get_conv_activation()
    anchors = model.get_anchors()
    onnx_name = model.get_onnx_name()

    with tempfile.TemporaryDirectory() as tensorboard_data_dir:
        print("Initializing data loaders")
        train_data_loader = torch.utils.data.DataLoader(
            ImageLabelDataset(train_uri, dataset_path=dataset_path, width=img_width, height=img_height, augment_hsv=augment_hsv,
                                augment_affine=augment_affine, num_images=num_train_images,
                                bw=bw, n_cpu=num_cpu, lr_flip=lr_flip, ud_flip=ud_flip,vis_batch=vis_batch,data_aug=data_aug,blur=blur,salt=salt,noise=noise,contrast=contrast,sharpen=sharpen,ts=ts,debug_mode=debug_mode, upload_dataset=upload_dataset),
            batch_size=(1 if debug_mode else batch_size),
            shuffle=(False if debug_mode else True),
            num_workers=(0 if vis_batch else num_cpu),
            pin_memory=cuda)
        print("Num train images: ", len(train_data_loader.dataset))

        validate_data_loader = torch.utils.data.DataLoader(
            ImageLabelDataset(validate_uri, dataset_path=dataset_path, width=img_width, height=img_height, augment_hsv=False,
                                augment_affine=False, num_images=num_validate_images,
                                bw=bw, n_cpu=num_cpu, lr_flip=False, ud_flip=False,vis_batch=vis_batch,data_aug=False,blur=False,salt=False,noise=False,contrast=False,sharpen=False,ts=ts,debug_mode=debug_mode, upload_dataset=upload_dataset),
            batch_size=(1 if debug_mode else batch_size),
            shuffle=False,
            num_workers=(0 if vis_batch else num_cpu),
            pin_memory=cuda)
        print("Num validate images: ", len(validate_data_loader.dataset))

        ##### additional configuration #####
        print("Training batch size: " + str(batch_size))
        
        print("Checkpoint interval: " + str(checkpoint_interval))

        print("Loss constants: " + str(loss_constant))

        print("Anchor boxes: " + str(anchors))

        print("Training image width: " + str(img_width))

        print("Training image height: " + str(img_height))

        print("Confidence Threshold: " + str(conf_thresh))

        print("Number of training classes: " + str(num_classes))

        print("Conv activation type: " + str(conv_activation))

        print("Starting learning rate: " + str(lr))

        if ts:
            print("Tile and scale mode [on]")
        else:
            print("Tile and scale mode [off]")

        if data_aug:
            print("Data augmentation mode [on]")
        else:
            print("Data augmentation mode [off]")

        ####################################

        start_epoch = 0

        weights_path = weights_path
        if optimizer_pick == "Adam":
            print("Using Adam Optimizer")
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=lr, weight_decay=weight_decay)
        elif optimizer_pick == "SGD":
            print("Using SGD Optimizer")
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise Exception(f"Invalid optimizer name: {optimizer_pick}")
        print("Loading weights")
        model.load_weights(weights_path, model.get_start_weight_dim())

        if torch.cuda.device_count() > 1:
            print('Using ', torch.cuda.device_count(), ' GPUs')
            model = nn.DataParallel(model)
        model = model.to(device, non_blocking=True)

        # Set scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

        val_loss = 999  # using a high number for validation loss
        val_loss_counter = 0
        step = [0]  # wrapping in an array so it is mutable
        epoch = start_epoch
        while epoch < num_epochs and step[0] < num_steps and not evaluate:
            epoch += 1
            scheduler.step()
            model.train()
            run_epoch(label_prefix="train", data_loader=train_data_loader, epoch=epoch,
                        step=step, model=model, num_epochs=num_epochs, num_steps=num_steps,
                        optimizer=optimizer)
            print('Completed epoch: ', epoch)
            # Update best loss
            if epoch % checkpoint_interval == 0 or epoch == num_epochs or step[0] >= num_steps:
                # First, save the weights
                save_weights_uri = os.path.join(output_uri, "{epoch}.weights".format(epoch=epoch))
                model.save_weights(save_weights_uri)

                with torch.no_grad():
                    print("Calculating loss on validate data")
                    epoch_losses, epoch_time_total, epoch_num_targets = run_epoch(
                        label_prefix="validate", data_loader=validate_data_loader, epoch=epoch,
                        model=model, num_epochs=num_epochs, num_steps=num_steps, optimizer=None,
                        step=step)
                    avg_epoch_loss = epoch_losses[0] / epoch_num_targets
                    print('Average Validation Loss: {0:10.6f}'.format(avg_epoch_loss))

                    if avg_epoch_loss > val_loss and epoch > min_epochs:
                        val_loss_counter += 1
                        print(f"Validation loss did not decrease for {val_loss_counter}"
                                f" consecutive check(s)")
                    else:
                        print("Validation loss decreased. Yay!!")
                        val_loss_counter = 0
                        val_loss = avg_epoch_loss
                        ##### updating best result for optuna study #####
                        result = open("logs/result.txt", "w" )
                        result.write(str(avg_epoch_loss))
                        result.close() 
                        ###########################################
                    validate.validate(dataloader=validate_data_loader, model=model, device=device, step=step[0], bbox_all=False,debug_mode=debug_mode)
                    if val_loss_counter == val_tolerance:
                        print("Validation loss stopped decreasing over the last " + str(val_tolerance) + " checkpoints, creating onnx file")
                        with tempfile.NamedTemporaryFile() as tmpfile:
                            model.save_weights(tmpfile.name)
                            weights_name = tmpfile.name
                            cfg_name = os.path.join(tempfile.gettempdir(), model_cfg.split('/')[-1].split('.')[0] + '.tmp')
                            onnx_gen = subprocess.call(['python3', 'yolo2onnx.py', '--cfg_name', cfg_name, '--weights_name', weights_name])
                            save_weights_uri = os.path.join(output_uri, onnx_name)
                            os.rename(weights_name, save_weights_uri)
                            try:
                                os.remove(onnx_name)
                            except:
                                pass
                            os.remove(cfg_name)
                        break
        if evaluate:
            validation = validate.validate(dataloader=validate_data_loader, model=model, device=device, step=-1, bbox_all=False, tensorboard_writer=None,debug_mode=debug_mode)
    return val_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    def add_bool_arg(name, default, help):
        arg_group = parser.add_mutually_exclusive_group(required=False)
        arg_group.add_argument('--' + name, dest=name, action='store_true', help=help)
        arg_group.add_argument('--no_' + name, dest=name, action='store_false', help=("Do not " + help))
        parser.set_defaults(**{name:default})

    parser.add_argument('--batch_size', type=int, default=7, help='size of each image batch')
    parser.add_argument('--optimizer_pick', type=str, default="Adam", help='choose optimizer between Adam and SGD')
    parser.add_argument('--model_cfg', type=str, help='cfg file path',required=True)
    parser.add_argument('--weights_path', type=str, help='initial weights path',required=True)
    parser.add_argument('--output_path', type=str, help='output weights path, by default we will create a folder based on current system time and name of your cfg file',default="automatic")
    parser.add_argument('--dataset_path', type=str, help='path to image dataset',default="dataset/YOLO_Dataset/")
    parser.add_argument('--num_epochs', type=int, default=2048, help='maximum number of epochs')
    parser.add_argument('--num_steps', type=int, default=8388608, help="maximum number of steps")
    parser.add_argument('--val_tolerance', type=int, default=3, help="tolerance for validation loss decreasing")
    parser.add_argument('--min_epochs', type=int, default=3, help="minimum training epochs")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    # Default output location of visualization is "Buckets/mit-dut-driverless-internal/dumping-ground/visualization/"
    parser.add_argument("--vis_batch", type=int, default=0, help="number of batches you wish to load and visualize before quitting training")

    ##### tile and scale #####
    add_bool_arg('ts', default=True, help="whether to initially scale the entire image by a constant factor determined by the appropriate cone pixel size, then chop and tile (instead of pad and resize)")
    ##########################

    add_bool_arg('augment_affine', default=False, help='whether to augment images')
    add_bool_arg('augment_hsv', default=False, help="whether to augment hsv")
    add_bool_arg('augment_lr_flip', default=False, help="whether to flip left/right")
    add_bool_arg('augment_ud_flip', default=False, help="whether to flip up/down")
    add_bool_arg('augment_blur', default=False, help="whether to add blur")
    add_bool_arg('augment_salt', default=False, help="whether to add salt/pepper")
    add_bool_arg('augment_noise', default=False, help="whether to add noise")
    add_bool_arg('augment_contrast', default=False, help="whether to add contrast")
    add_bool_arg('augment_sharpen', default=False, help="whether to add sharpen")
    add_bool_arg('evaluate', default =False, help="If we want to get the mAP values rather than train")
    ##########################

    add_bool_arg('vanilla_anchor', default=False, help="whether to use vanilla anchor boxes for training")
    add_bool_arg('debug_mode', default=False, help="whether to visualize the validate prediction during mAP calculation, need to make CUDA=False at first. If true then batch size will also automatically set to 1 and training shuffle will be False. ")
    add_bool_arg('data_aug', default=False, help="whether to do all stable data augmentation")
    add_bool_arg('upload_dataset', default=False, help="whether to uploading all tiles to GCP, have to enable --ts first")

    
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for the optimizer')
    parser.add_argument('--gamma', type=float, default=0.95, help='gamma for the scheduler')
    parser.add_argument('--lr', type=float, default=0.001, help='starting learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')

    ##### Loss Constants #####
    parser.add_argument('--xy_loss', type=float, default=2, help='confidence loss for x and y')
    parser.add_argument('--wh_loss', type=float, default=1.6, help='confidence loss for width and height')
    parser.add_argument('--no_object_loss', type=float, default=25, help='confidence loss for non-objectness')
    parser.add_argument('--object_loss', type=float, default=0.1, help='confidence loss for objectness')
    
    opt = parser.parse_args()
   
    save_file_name = 'logs/' + opt.model_cfg.split('/')[-1].split('.')[0]
    sys.stdout = Logger(save_file_name + '.log')
    sys.stderr = Logger(save_file_name + '.error')

    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, save_file_name.split('/')[-1] + '.tmp')
    shutil.copy2(opt.model_cfg, temp_path)
    label = subprocess.call(["git", "describe", "--always"])
    result = main(evaluate=opt.evaluate,
                  batch_size=opt.batch_size,
                  optimizer_pick=opt.optimizer_pick,
                  model_cfg=opt.model_cfg,
                  weights_path=opt.weights_path,
                  output_path=opt.output_path,
                  dataset_path=opt.dataset_path,
                  num_epochs=opt.num_epochs,
                  num_steps=(opt.num_steps if opt.vis_batch is 0 else opt.vis_batch),
                  checkpoint_interval=opt.checkpoint_interval,
                  augment_affine=opt.augment_affine,
                  augment_hsv=opt.augment_hsv,
                  lr_flip=opt.augment_lr_flip,
                  ud_flip=opt.augment_ud_flip,
                  momentum=opt.momentum,
                  gamma=opt.gamma,
                  lr=opt.lr,
                  weight_decay=opt.weight_decay,
                  vis_batch=opt.vis_batch,
                  data_aug=opt.data_aug,
                  blur=opt.augment_blur,
                  salt=opt.augment_salt,
                  noise=opt.augment_noise,
                  contrast=opt.augment_contrast,
                  sharpen=opt.augment_sharpen,
                  ts=opt.ts,
                  debug_mode=opt.debug_mode,
                  upload_dataset=opt.upload_dataset,
                  xy_loss=opt.xy_loss,
                  wh_loss=opt.wh_loss,
                  no_object_loss=opt.no_object_loss,
                  object_loss=opt.object_loss,
                  vanilla_anchor=opt.vanilla_anchor,
                  val_tolerance=opt.val_tolerance,
                  min_epochs=opt.min_epochs
                  )
