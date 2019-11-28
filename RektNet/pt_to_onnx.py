import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch import nn
from keypoint_net import KeypointNet

import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch import nn

import argparse
import os
import sys


def main(weights_uri,onnx_name):
    model = KeypointNet(7, (80, 80),onnx_mode=True)

    weights_path = weights_uri

    model.load_state_dict(torch.load(weights_path, map_location='cpu').get('model'))
    torch.onnx.export(model, torch.randn(1, 3, 80, 80), onnx_name)

    print("onnx file conversion succeed and saved at: " + onnx_name)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='.pt weights file convert to .onnx')
    parser.add_argument('--onnx_name', default='new_keypoints.onnx',
                                            help='the name of output onnx file')

    parser.add_argument('--weights_uri', required=True,
                                            help='Path to weights file')
    args = parser.parse_args()
    

    main(weights_uri=args.weights_uri, onnx_name=args.onnx_name)
