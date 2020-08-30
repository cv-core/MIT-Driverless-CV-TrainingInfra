### Description

This is our custom Key Points detection network

## Requirements:

* CUDA>=10.1
* python==3.6
* opencv_python==4.1.0.25
* numpy==1.16.4
* torch==1.1.0
* torchvision==0.3.0
* pandas==0.24.2
* optuna==0.19.0
* Pillow==6.2.1
* protobuf==3.11.0
* pymysql==0.9.3
* tqdm==4.39.0

## Usage
### 1.Download our dataset

##### Download through GCP Tookit
###### 1.1 Image dataset:
```
gsutil cp -p gs://mit-driverless-open-source/RektNet_Dataset.zip ./dataset/
```
then unzip 
```
unzip dataset/RektNet_Dataset.zip -d ./dataset/
```
###### 1.2 Label csv file:
```
gsutil cp -p gs://mit-driverless-open-source/rektnet-training/rektnet_label.csv ./dataset/
```

##### Download manually (Optional)
You can download image dataset and label csv from the link below and unzip them into `./dataset/RektNet_Dataset/` 

[Image dataset](https://storage.cloud.google.com/mit-driverless-open-source/RektNet_Dataset.zip?authuser=1)

[All label csv](https://storage.cloud.google.com/mit-driverless-open-source/rektnet-training/rektnet_label.csv?authuser=1)

### 2.Training

```
python3 train_eval.py --study_name=<name for this experiment>
```

Once you've finished training, you can access the weights file in `./outputs/`

### 3.Inference

#### To download our pretrained Keypoints weights for *Formula Student Standard*, click ***[here](https://storage.googleapis.com/mit-driverless-open-source/pretrained_kpt.pt)***


```
python3 detect.py --model=<path to .pt weights file> --img=<path to an image>
```

Once you've finished inference, you can access the result in `./outputs/visualization/`

#### Run Bayesian hyperparameter search

Before running the Bayesian hyperparameter search, make sure you know what specific hyperparameter that you wish to tuning on, and a reasonable operating range/options of that hyperparameter.

Go into the `objective()` function of `train_hyper.py` edit your custom search

Then launch your Bayesian hyperparameter search
```
python3 train_eval_hyper.py --study_name=<give it a proper name>
```

#### Convert .weights to .onnx manually

Though our training scrip will do automatical .pt->.onnx conversion, you can always do it manually
```
python3 yolo2onnx.py --onnx_name=<path to output .onnx file> --weights_uri=<path to your .pt file>
```