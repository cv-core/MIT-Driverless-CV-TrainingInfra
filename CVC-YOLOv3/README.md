### Description

The repo is originally forked from https://github.com/ultralytics/yolov3 and contains inference and training code for YOLOv3 in PyTorch.

## Requirements:

* CUDA >= 10.1
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

#### 1.1 Environment Setup (Optional)

```
sudo python3 setup.py build develop
```

### 2.Training

```
python3 train.py --model_cfg=model_cfg/yolo_baseline.cfg --weights_path=dataset/sample-yolov3.weights
```

Once you've finished training, you can access the weights file in `./outputs/`

### 3.Inference

```
python3 detect.py --model_cfg=model_cfg/yolo_baseline.cfg --target_path=<path to an image or video> --weights_path=<path to your trained weights file>
```

Once you've finished inference, you can access the result in `./outputs/visualization/`

#### Run Bayesian hyperparameter search

Before running the Bayesian hyperparameter search, make sure you know what specific hyperparameter that you wish to tuning on, and a reasonable operating range/options of that hyperparameter.

Go into the `objective()` function of `train_hyper.py` edit your custom search

Then launch your Bayesian hyperparameter search
```
python3 train_hyper.py --model_cfg=<path to cfg file> --study_name=<give it a proper name>
```

#### Splits your own csv file 

```
python3 generate_kmeans_dataset_csvs.py --input_csvs=<path to your csv file that contains all the label> --dataset_path=<path to your image dataset>
```

