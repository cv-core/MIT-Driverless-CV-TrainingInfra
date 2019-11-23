import torch
import cv2
import numpy as np
import argparse
import sys
import os
import sys
import shutil
from utils import vis_tensor_and_up2gcp, prep_image

sys.path.insert(1, os.path.realpath(os.path.pardir+'/vectorized_yolov3/utils'))
import storage_client
from keypoint_net import KeypointNet

detection_tmp_path = "/tmp/detect/"

if os.path.exists(detection_tmp_path):
    shutil.rmtree(detection_tmp_path)  # delete output folder
os.makedirs(detection_tmp_path)  # make new output folder

def download_file(file_uri):
    os_filepath = storage_client.get_file(file_uri)
    if not os.path.isfile(os_filepath):
        raise Exception("could not download image: {file_uri}".format(file_uri=file_uri))
    return os_filepath

def main(model,img,img_size,output,flip,rotate):

    output_path = output

    model_path = model
    if model_path.startswith('gs'):
        model_filepath = download_file(model_path)
    else:
        model_filepath = model_path

    image_path = img

    if image_path.startswith('gs'):
        image_filepath = download_file(image_path)
    else:
        image_filepath = image_path

    img_name = '_'.join(image_filepath.split('/')[-1].split('.')[0].split('_')[-5:])

    image_size = (img_size, img_size)

    image = cv2.imread(image_filepath)
    h, w, _ = image.shape

    image = prep_image(image=image,target_image_size=image_size)
    image = (image.transpose((2, 0, 1)) / 255.0)[np.newaxis, :]
    image = torch.from_numpy(image).type('torch.FloatTensor')

    model = KeypointNet()
    model.load_state_dict(torch.load(model_filepath).get('model'))
    model.eval()
    output = model(image)
    out = np.empty(shape=(0, output[0][0].shape[2]))
    for o in output[0][0]:
        chan = np.array(o.cpu().data)
        cmin = chan.min()
        cmax = chan.max()
        chan -= cmin
        chan /= cmax - cmin
        out = np.concatenate((out, chan), axis=0)
    cv2.imwrite(detection_tmp_path + img_name + "_hm.jpg", out * 255)
    storage_client.upload_file(detection_tmp_path + img_name + "_hm.jpg", output_path + img_name + "_hm.jpg")


    image = cv2.imread(image_filepath)
    h, w, _ = image.shape

    vis_tensor_and_up2gcp(image=image, h=h, w=w, tensor_output=output[1][0].cpu().data, image_name=img_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Keypoints Visualization')
    def add_bool_arg(name, default, help):
        arg_group = parser.add_mutually_exclusive_group(required=False)
        arg_group.add_argument('--' + name, dest=name, action='store_true', help=help)
        arg_group.add_argument('--no_' + name, dest=name, action='store_false', help=("Do not " + help))
        parser.set_defaults(**{name:default})

    parser.add_argument('--model', help='path to model', type=str, required=True)
    parser.add_argument('--img', help='path to single image', type=str, default="gs://mit-dut-driverless-external/ConeColourLabels/vid_3_frame_22063_0.jpg")
    parser.add_argument('--img_size', help='image size', default=80, type=int)
    parser.add_argument('--output', help='path to upload the detection', default="gs://mit-dut-driverless-internal/dumping-ground/keypoints_visualization/")

    add_bool_arg('flip', default=False, help='flip image')
    add_bool_arg('rotate', default=False, help='rotate image')

    args = parser.parse_args(sys.argv[1:])

    main(model=args.model,img=args.img,img_size=args.img_size,output=args.output,flip=args.flip,rotate=args.rotate)
