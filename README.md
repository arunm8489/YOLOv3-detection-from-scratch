# YOLOv3-detection-from-scratch
This is a minimal implementation of YOLOv3 from scratch using PyTorch 


## Requirements
* Python 3.6
* Pytorch 1.3.1
* Open-cv

## Usage

First you have to downoad the weights using

'''
wget https://pjreddie.com/media/files/yolov3.weights 
'''

Now you can detect images using

'''
python detect.py --input_image imagename --claasses classfile
'''

Output will be saved as 'detected_' + imagename


## Examples


Try this markdown:

![image](https://github.com/arunm8489/YOLOv3-detection-from-scratch/blob/master/messi.jpg)
![detected image](https://github.com/arunm8489/YOLOv3-detection-from-scratch/blob/master/detected_messi.jpg)




