# YOLOv3-detection-from-scratch
This is a minimal implementation of YOLOv3 from scratch using PyTorch 


## Requirements
* Python 3.6
* Pytorch 
* Open-cv

## Usage

First you have to downoad the weights using

wget https://pjreddie.com/media/files/yolov3.weights 

Now ypu can detect images using

python detect.py --input_image imagename --claasses classfile

Output will be saved as 'detected_' + imagename


## Examples

