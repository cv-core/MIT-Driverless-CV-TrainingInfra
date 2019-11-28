from __future__ import division

import torch
import torch.nn as nn
import numpy as np
import os
from datetime import datetime
import csv

from utils.parse_config import parse_model_config
from utils.utils import build_targets

vanilla_anchor_list = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]

def create_modules(module_defs,xy_loss,wh_loss,no_object_loss,object_loss,vanilla_anchor):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    img_width = int(hyperparams["width"])
    img_height = int(hyperparams["height"])
    onnx_height = int(hyperparams["onnx_height"])
    num_classes = int(hyperparams["classes"])
    leaky_slope = float(hyperparams["leaky_slope"])
    conv_activation = hyperparams["conv_activation"]
    yolo_masks = [[int(y) for y in x.split(',')] for x in hyperparams["yolo_masks"].split('|')]
    ##### reading anchors from train.csv #####
    csv_uri = hyperparams["train_uri"]
    training_csv_tempfile = csv_uri
    with open(training_csv_tempfile) as f:
        csv_reader = csv.reader(f)
        row = next(csv_reader)
        row = str(row)[2:-2]
        anchor_list = [[float(y) for y in x.split(',')] for x in row.split("'")[0].split('|')]
    #############################

    ##### using vanilla anchor boxes if its switch is on #####
    if vanilla_anchor:
        anchor_list = vanilla_anchor_list
    #############################
    build_targets_ignore_thresh=float(hyperparams["build_targets_ignore_thresh"])
    module_list = nn.ModuleList()
    
    yolo_count = 0
    act_flag = 1 #all pre yolo layers need linear activations
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()
        if module_def["type"] == "convolutional":
            bn = 1
            if module_def["filters"] == 'preyolo':
                filters = (num_classes + 5) * len(yolo_masks[yolo_count])
                act_flag = 0
                bn = 0
            else:
                filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = int((kernel_size - 1) // 2)
            modules.add_module("conv_%d" % i, nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn))
            if bn:
                modules.add_module("batch_norm_%d" % i, nn.BatchNorm2d(filters))
            if conv_activation == "leaky" and act_flag == 1:
                modules.add_module("leaky_%d" % i, nn.LeakyReLU(leaky_slope))
            if conv_activation == "ReLU" and act_flag == 1:
                modules.add_module("ReLU_%d" % i, nn.ReLU()) 
            act_flag = 1

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                padding = nn.ZeroPad2d((0, 1, 0, 1))
                modules.add_module("_debug_padding_%d" % i, padding)
            maxpool = nn.MaxPool2d(
                kernel_size=int(module_def["size"]),
                stride=int(module_def["stride"]),
                padding=int((kernel_size - 1) // 2))
            modules.add_module("maxpool_%d" % i, maxpool)

        elif module_def["type"] == "upsample":
            upsample = nn.Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module("upsample_%d" % i, upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = 0
            for layer_i in layers:
                if layer_i > 0:
                    layer_i += 1
                filters += output_filters[layer_i]
            modules.add_module("route_%d" % i, EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = output_filters[int(module_def["from"])]
            modules.add_module("shortcut_%d" % i, EmptyLayer())

        elif module_def["type"] == "yolo":
            anchors = ([anchor_list[i] for i in yolo_masks[yolo_count]])
            yolo_layer = YOLOLayer(anchors, num_classes, img_height, img_width, build_targets_ignore_thresh,conv_activation,xy_loss,wh_loss,object_loss,no_object_loss)
            modules.add_module("yolo_%d" % i, yolo_layer)
            yolo_count += 1
        module_list.append(modules)
        output_filters.append(filters)
    return hyperparams, module_list

class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_height, img_width, build_targets_ignore_thresh, conv_activation, xy_loss, wh_loss, object_loss, no_object_loss):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.image_height = img_height
        self.image_width = img_width
        self.ignore_thres = build_targets_ignore_thresh
        self.xy_loss = xy_loss
        self.wh_loss = wh_loss
        self.no_object_loss = no_object_loss
        self.object_loss = object_loss
        self.conv_activation = conv_activation

        self.mse_loss = nn.MSELoss(size_average=True)  # Coordinate loss
        self.bce_loss = nn.BCELoss(size_average=True)  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()  # Class loss

    def forward(self, sample, targets=None):
        nA = self.num_anchors
        nB = sample.size(0)
        nGh = sample.size(2)
        nGw = sample.size(3)
        stride = self.image_height / nGh
        
        prediction = sample.view(nB, nA, self.bbox_attrs, nGh, nGw).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # Calculate offsets for each grid
        grid_x = torch.arange(nGw, dtype=torch.float, device=x.device).repeat(nGh, 1).view([1, 1, nGh, nGw])
        grid_y = torch.arange(nGh, dtype=torch.float, device=x.device).repeat(nGw, 1).t().view([1, 1, nGh, nGw]).contiguous()
        scaled_anchors = torch.tensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors], dtype=torch.float, device=x.device)
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

        # Add offset and scale with anchors
        pred_boxes = torch.zeros(prediction[..., :4].shape, dtype=torch.float, device=x.device)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
        
        # Training
        if targets is not None:
            self.mse_loss = self.mse_loss.to(sample.device, non_blocking=True)
            self.bce_loss = self.bce_loss.to(sample.device, non_blocking=True)
            self.ce_loss = self.ce_loss.to(x.device, non_blocking=True)
            mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
                target=targets,
                anchors=scaled_anchors,
                num_anchors=nA,
                num_classes=self.num_classes,
                grid_size_h=nGh,
                grid_size_w=nGw,
                ignore_thres=self.ignore_thres,
            )

            # Handle target variables
            tx.requires_grad_(False)
            ty.requires_grad_(False)
            tw.requires_grad_(False)
            th.requires_grad_(False)
            tconf.requires_grad_(False)
            tcls.requires_grad_(False)

            # Get conf mask where gt and where there is no gt
            conf_mask_true = mask
            conf_mask_false = conf_mask - mask

            # Mask outputs to ignore non-existing objects
            loss_x = self.xy_loss * self.mse_loss(x[mask], tx[mask])
            loss_y = self.xy_loss * self.mse_loss(y[mask], ty[mask])
            loss_w = self.wh_loss * self.mse_loss(w[mask], tw[mask])
            loss_h = self.wh_loss * self.mse_loss(h[mask], th[mask])
            #We are only doing single class detection, so we set loss_cls always to be 0. You can always make it to another value if you wish to do multi-class training
            loss_cls_constant = 0
            loss_cls = loss_cls_constant * (1 / nB) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
            
            loss_noobj = self.no_object_loss * self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) 
            loss_obj = self.object_loss * self.bce_loss(pred_conf[conf_mask_true], tconf[conf_mask_true])
            loss = loss_x + loss_y + loss_w + loss_h + loss_noobj + loss_obj + loss_cls

            return loss, torch.tensor((loss_x, loss_y, loss_w, loss_h, loss_obj, loss_noobj), device=targets.device)

        else:
            # If not in training phase return predictions
            output = torch.cat((
                    pred_boxes.view(nB, -1, 4) * stride,
                    pred_conf.view(nB, -1, 1),
                    pred_cls.view(nB, -1, self.num_classes)),
                    -1)
            return output

class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, xy_loss, wh_loss, no_object_loss, object_loss,vanilla_anchor):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        
        self.hyperparams, self.module_list = create_modules(module_defs=self.module_defs,xy_loss=xy_loss,wh_loss=wh_loss,no_object_loss=no_object_loss,object_loss=object_loss,vanilla_anchor=vanilla_anchor)
        self.img_width = int(self.hyperparams["width"])
        self.img_height = int(self.hyperparams["height"])
        # in order to help train.py defines the onnx filename since it is not defined by yolo2onnx.py
        self.onnx_height = int(self.hyperparams["onnx_height"])
        self.onnx_name = config_path.split('/')[-1].split('.')[0] + '_' + str(self.img_width) + str(self.onnx_height) + '.onnx'
        self.num_classes = int(self.hyperparams["classes"])
        if int(self.hyperparams["channels"]) == 1:
            self.bw = True
        elif int(self.hyperparams["channels"]) == 3:
            self.bw = False
        else:
            print('Channels in cfg file is not set properly, making it colour')
            self.bw = False
        current_month = datetime.now().strftime('%B').lower()
        current_year = str(datetime.now().year)

        self.validate_uri = self.hyperparams["validate_uri"]
        self.train_uri = self.hyperparams["train_uri"]
        self.num_train_images = int(self.hyperparams["num_train_images"])
        self.num_validate_images = int(self.hyperparams["num_validate_images"])
        self.conf_thresh = float(self.hyperparams["conf_thresh"])
        self.nms_thresh = float(self.hyperparams["nms_thresh"])
        self.iou_thresh = float(self.hyperparams["iou_thresh"])
        self.start_weights_dim = [int(x) for x in self.hyperparams["start_weights_dim"].split(',')]
        self.conv_activation = self.hyperparams["conv_activation"]

        ##### loss constants #####
        self.xy_loss=xy_loss
        self.wh_loss=wh_loss
        self.no_object_loss=no_object_loss
        self.object_loss=object_loss
        ##### reading anchors from train.csv #####
        csv_uri = self.hyperparams["train_uri"]
        training_csv_tempfile = csv_uri
        with open(training_csv_tempfile) as f:
            csv_reader = csv.reader(f)
            row = next(csv_reader)
            row = str(row)[2:-2]
            anchor_list = [[float(y) for y in x.split(',')] for x in row.split("'")[0].split('|')]
        #############################

        ##### using vanilla anchor boxes until skanda dataloader is done #####
        if vanilla_anchor:
            anchor_list = vanilla_anchor_list
        #############################
        self.anchors = anchor_list
        self.seen = 0
        self.header_info = torch.tensor([0, 0, 0, self.seen, 0])

    def get_start_weight_dim(self):
        return self.start_weights_dim

    def get_onnx_name(self):
        return self.onnx_name

    def get_bw(self):
        return self.bw
    
    def get_loss_constant(self):
        return [self.xy_loss,self.wh_loss,self.no_object_loss,self.object_loss]

    def get_conv_activation(self):
        return self.conv_activation

    def get_num_classes(self):
        return self.num_classes

    def get_anchors(self):
        return self.anchors

    def get_threshs(self):
        return self.conf_thresh, self.nms_thresh, self.iou_thresh
    
    def img_size(self):
        return self.img_width, self.img_height
    
    def get_links(self):
        return self.validate_uri, self.train_uri
    
    def num_images(self):
        return self.num_validate_images, self.num_train_images
    
    def forward(self, x, targets=None):
        is_training = targets is not None
        output = []

        if is_training:
            total_losses = torch.zeros(6, device=targets.device)
        layer_outputs = []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                layer_i = [int(x) for x in module_def["layers"].split(",")]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                # Train phase: get loss
                if is_training:
                    x, losses = module[0](x, targets)
                    total_losses += losses
                # Test phase: Get detections
                else:
                    x = module(x)
                output.append(x)
            layer_outputs.append(x)
        return (sum(output), *total_losses) if is_training else torch.cat(output, 1)
    def load_weights(self, weights_path, start_weight_dim):
        # Open the weights file
        fp = open(weights_path, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values
        # Needed to write header when saving weights
        self.header_info = header

        self.seen = header[3]
        weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
        fp.close()

        ptr = 0
        yolo_count = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["filters"] != 'preyolo':
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                    # Load conv. weights
                    num_w = conv_layer.weight.numel()
                    conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                    conv_layer.weight.data.copy_(conv_w)
                    ptr += num_w
                elif module_def["filters"] == 'preyolo':
                    orig_dim = start_weight_dim[yolo_count]
                    yolo_count += 1
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += orig_dim
                    # Load conv. weights
                    num_w = conv_layer.weight.numel()
                    dummyDims = [orig_dim] + list(conv_layer.weight.size()[1:])
                    dummy = torch.zeros(tuple(dummyDims))
                    conv_w = torch.from_numpy(weights[ptr : ptr + int(num_w * orig_dim / num_b)]).view_as(dummy)
                    conv_w = conv_w[0:num_b][:][:][:]
                    conv_layer.weight.data.copy_(conv_w)
                    ptr += int(num_w * orig_dim / num_b)
                else:
                    print(module)
                    raise Exception('The above layer has its BN or preyolo defined wrong')

    def save_weights(self, path, cutoff=-1):

        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["filters"] != 'preyolo':
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()
