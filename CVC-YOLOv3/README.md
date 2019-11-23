### Description

The repo is forked from https://github.com/ultralytics/yolov3 and contains inference and training code for YOLOv3 in PyTorch.

### Setup
Make sure to have followed `setup-gcloud.md` and be logged into the corresponding GCloud instance. In particular, make sure to have run the `setup.py` command as directed there.

Then run the following, note that the generate data script has to have links that are the gs: version and not the 'storage.api' version:
```
source /home/mit-dut-driverless-internal/venvs/cv/bin/activate
# log in through the browser on your local machine
gcloud auth application-default login 
# Note that the csv_uri must have at least 100 images in it and it must have the gs: locations referenced rather than their local locations
python generate_dataset_csvs.py --csv_uris \
	gs://mit-dut-driverless-internal/data-labels/fullFrameTest.csv \
	--output_uri gs://mit-dut-driverless-internal/vectorized-yolov3-training/january-experiments/fullFrameNonSquareTest/

python train.py --model_cfg model_cfgs/yolov3_80class.cfg \
	--validate_uri gs://mit-dut-driverless-internal/vectorized-yolov3-training/hive-0-1-2-test.csv \
	--train_uri gs://mit-dut-driverless-internal/vectorized-yolov3-training/hive-0-1-2-train.csv \
	--output_uri gs://mit-dut-driverless-internal/vectorized-yolov3-training/quickstart/ \
	--study_name color_80class \
	--weights_uri gs://mit-dut-driverless-internal/vectorized-yolov3-training/december-experiments/92da791a96de485895ea219f7035c2aa/36.weights 2>&1 | tee results/color_416_baseline.log
```

The same command in one line:
```
python train.py --model_cfg model_cfgs/yolov3_80class.cfg --validate_uri gs://mit-dut-driverless-internal/vectorized-yolov3-training/hive-0-1-2-test.csv --train_uri gs://mit-dut-driverless-internal/vectorized-yolov3-training/hive-0-1-2-train.csv --output_uri gs://mit-dut-driverless-internal/vectorized-yolov3-training/january-experiments/color_416_baseline --study_name color_416_baseline 2>&1 | tee logs/color_416_baseline.log
```

Back-up the logs:
```
gsutil cp logs/color_416_baseline.log gs://mit-dut-driverless-internal/vectorized-yolov3-training/january-experiments/color_416_baseline/color_416_baseline.log
```

To run detection on one image, and visualize the results:
```
python detect.py --weights_uri gs://mit-dut-driverless-internal/vectorized-yolov3-training/sample-yolov3.weights --image_uri gs://mit-dut-driverless-external/HiveAIRound2/vid_38_frame_956.jpg --output_uri gs://mit-dut-driverless-internal/vectorized-yolov3-training/january-experiments/fullFrameNonSquareTest2/vid38f956.jpg --img_width 2048 --img_height 1536 --model_cfg model_cfgs/yolov3_80class_fullFrame.cfg
```
To run detection on multiple images of your choosing, write a bash file similar to:
```
run_scripts/run_detect.sh
```
And to copy the visualizations to a local machine (with GCloud Client installed): `gsutil -m cp -r gs://mit-dut-driverless-internal/vectorized-yolov3-training/january-experiments/color_416_baseline/visualization .`

You can also add the following to your ~/.bashrc to make things easier:
```
cd /home/mit-dut-driverless-internal/cv-cone-town/vectorized-yolov3/
source ../../venvs/cv/bin/activate
```

You can create a video from a list of extracted frames from a video using:
```
run_scripts/make_video.sh
```
You'll need to adjust the hardcoded paths at the top of the file.

To-Do:
Splits:
    - Get splits working with online processing, re-train models
    - Understand why splits work so well
    - Add splits to Xavier processing module
- Make experiment folders readable

### Bookmarked Models
Color Full-Frame: gs://mit-dut-driverless-internal/vectorized-yolov3-training/december-experiments/92da791a96de485895ea219f7035c2aa/36.weights
Black and White Full-Frame: gs://mit-dut-driverless-internal/vectorized-yolov3-training/december-experiments/2c5485a8ee6847808459f54cac50ae8e/64.weights

Color Split-Frame: gs://mit-dut-driverless-internal/vectorized-yolov3-training/december-experiments/56070f8a1e454b2383c12d0fec37e3dc/104.weights

Black and White Split-Frame: gs://mit-dut-driverless-internal/vectorized-yolov3-training/december-experiments/701e0c805b4d4052a1798a4d9c3c5914/68.weights

