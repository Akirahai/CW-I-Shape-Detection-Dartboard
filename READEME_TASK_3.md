# Using Darknet to train YoLO
## Darknet
- Open source neural network framework written in C and CUDA. It is fasst, easy to install and supports CPU and GPU computation. 
- Clonde Darknet `git clone https://github.com/AlexeyAB/darknet`.
- Run on CPU only as I only have CPU (no GPU version)

### Requirements
- Having wsl `wsl`
```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y python3-dev python3-pip python3-venv
sudo apt install -y libopencv-dev
sudo apt install -y build-essential cmake pkg-config
```
- Create python environment
```bash
python3 -m venv myenv
source myenv/bin/activate
pip install --upgrade pip
```
- Install Necessary Library
```bash
pip install -r requirements.txt
git clone https://github.com/AlexeyAB/darknet
```


## Quick Start
- Configure the darknet library.
```bash
cd darknet
sed -i 's/OPENCV=0/OPENCV=1/' Makefile
make
```
- Download YoLo Weights in the main directory:
```bash
cd ..
!wget - quiet https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
```
- Test the model using dart0.jpg(copy into the data folder)
```bash
cd darknet
./darknet detector test cfg/coco.data cfg/yolov4.cfg ../yolov4.weights data/dart0.jpg -dont_show
```
- We got the following results:
```bash
Done! Loaded 162 layers from weights-file 
 Detection layer: 139 - type = 28 
 Detection layer: 150 - type = 28
 Detection layer: 161 - type = 28
data/dart0.jpg: Predicted in 10374.103000 milli-seconds.
person: 96%
microwave: 32%
```
- As the model dataset does not have object as dartboard, so the dartboard was mistaken with microwave/ umbrella.

## Prepare the dataset:
- Move the dataset (include data augmentation picture and labels) into `darknet/data/obj`:
    - `*.jpg`: the image picture
    - `*.txt`: the image labels.
- Download the `yolov4-tiny` weights for this dataset:
```bash
wget - quiet https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29
cp yolov4-tiny.conv.29 darknet/build/darknet/x64/
```
- Rename the config file that in charge of configuring the yolov4tiny architecture:
```bash
cd darknet
# copy the base config for tiny
cp cfg/yolov4-tiny-custom.cfg cfg/yolov4-tiny-dartboard.cfg

# set max_batches — for a small custom dataset you might set something like 4000
sed -i 's/max_batches = 500200/max_batches = 4000/' cfg/yolov4-tiny-dartboard.cfg

# adjust subdivisions to fit your GPU / memory
sed -i 's/subdivisions=1/subdivisions=16/' cfg/yolov4-tiny-dartboard.cfg

# adjust learning-rate schedule (steps) relative to max_batches
sed -i 's/steps=400000,450000/steps=3200,3600/' cfg/yolov4-tiny-dartboard.cfg

# change number of classes = 1 (since only dartboard)
sed -i 's/classes=80/classes=1/g' cfg/yolov4-tiny-dartboard.cfg

# update the number of filters in the convolution layers preceding each YOLO layer
# filters formula: filters = (classes + 5) * 3 = (1 + 5) * 3 = 18
sed -i 's/filters=255/filters=18/g' cfg/yolov4-tiny-dartboard.cfg
# likewise for any other filter lines (if more than one)
sed -i 's/filters=57/filters=18/g' cfg/yolov4-tiny-dartboard.cfg
```
- Train the model:
```bash
./darknet/darknet detector train darknet/data/obj.data darknet/cfg/yolov4-tiny-dartboard.cfg yolov4-tiny.conv.29 -dont_show -map
```
- Run the model:
```python
from torch_snippets import Glob, stem, show, read

image_paths = Glob('images-of-dartboard')

for f in image_paths:
    !./darknet detector test \
    darknet/data/obj.data \
    darknet/cfg/yolov4-tiny-dartboard.cfg \
    darknet/backup/yolov4-tiny-dartboard_4000.weights \
    {f}

    !mv predictions.jpg {stem(f)}_pred.jpg

```