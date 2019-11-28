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
