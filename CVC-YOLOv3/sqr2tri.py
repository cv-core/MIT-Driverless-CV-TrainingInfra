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
import statistics
import torch
import csv
import argparse
from tqdm import tqdm

from utils import storage_client

import PIL
from PIL import Image, ImageDraw

def main(csv_uri,output_uri):
    print(csv_uri)
    tmpFile = storage_client.get_file(csv_uri)
    train_data_table = pd.read_csv(tmpFile)
    train_images, train_labels = None, None

    if train_labels is None:
        # Separate the labels from the input data.
        images = train_data_table.values[:, 1]
        labels = train_data_table.values[:, 2:]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    def add_bool_arg(name, default, help):
        arg_group = parser.add_mutually_exclusive_group(required=False)
        arg_group.add_argument('--' + name, dest=name, action='store_true', help=help)
        arg_group.add_argument('--no_' + name, dest=name, action='store_false', help=("Do not " + help))
        parser.set_defaults(**{name:default})

    parser.add_argument("--input_csvs", help="csv file to split", default = 'gs://mit-dut-driverless-internal/data-labels/Hive/hive-ai-round-0123-gs-uris.csv')
    parser.add_argument("--output_bucket", type=str, help="Folder name to plop all the data", default = 'gs://mit-dut-driverless-internal/vectorized-yolov3-training/helpers/hive0123/')
    opt = parser.parse_args()

    main(csv_uri=opt.input_csvs,
    output_uri=opt.output_bucket)