### Description

The repo is originally forked from https://github.com/ultralytics/yolov3 and contains inference and training code for YOLOv3 in PyTorch.

## Requirements:

* python=3.6
* numpy==1.16.4
* matplotlib==3.1.0
* torchvision==0.3.0
* opencv_python==4.1.0.25
* torch==1.1.0
* requests==2.20.0
* pandas==0.24.2
* imgaug==0.3.0
* onnx==1.6.0
* optuna==0.19.0
* Pillow==6.2.1
* protobuf==3.11.0
* pymysql==0.9.3
* retrying==1.3.3
* tensorboardX==1.9
* tqdm==4.39.0

## Usage
### 1.Download our dataset

##### Download through GCP Tookit
Image dataset:
```
gsutil cp -p gs://mit-driverless-open-source/YOLO_Dataset.zip ./dataset/
```
then unzip 
```
unzip dataset/YOLO_Dataset.zip
```
Label csv file:
```
gsutil cp -p gs://mit-driverless-open-source/yolov3-training/all.csv ./dataset/
```
```
gsutil cp -p gs://mit-driverless-open-source/yolov3-training/train.csv ./dataset/
```
```
gsutil cp -p gs://mit-driverless-open-source/yolov3-training/validate.csv ./dataset/
```
Initial weights file:
```
gsutil cp -p  gs://mit-driverless-open-source/yolov3-training/sample-yolov3.weights ./dataset/
```

##### Download manually (Optional)
You can download image dataset and label csv from the link below and unzip them into `./dataset/YOLO_Dataset/` 

[Image dataset](https://storage.cloud.google.com/mit-driverless-open-source/YOLO_Dataset.zip?authuser=1)

[All label csv](https://storage.cloud.google.com/mit-driverless-open-source/yolov3-training/all.csv?authuser=1)

[Train label csv](https://storage.cloud.google.com/mit-driverless-open-source/yolov3-training/train.csv?authuser=1)

[Validate label csv](https://storage.cloud.google.com/mit-driverless-open-source/yolov3-training/validate.csv?authuser=1)

[Initial weights file](https://storage.cloud.google.com/mit-driverless-open-source/yolov3-training/sample-yolov3.weights?authuser=1)

2. Train the model with

```
python train.py
```
