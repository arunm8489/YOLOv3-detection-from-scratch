# YOLOv3-detection-from-scratch
This is a minimal implementation of YOLOv3 from scratch using PyTorch 

### Tutorials
You can refer my tutorials to check how I implemented it from scratch (https://medium.com/@arunm8489)
* YOLOv3 From Scratch Using PyTorch(Part1)
* YOLOv3 From Scratch Using PyTorch(Part2)



## Requirements
* Python 3.6
* Pytorch 1.3.1
* Open-cv

## Usage

First you have to downoad the yolov3 weights using

```
wget https://pjreddie.com/media/files/yolov3.weights 
```

Now you can detect images using

```
python detect.py --input_image imagename --classes classfile
```

Output will be saved as 'detected_' + imagename


## Examples


Try this markdown:

![image](https://github.com/arunm8489/YOLOv3-detection-from-scratch/blob/master/messi.jpg)
![detected image](https://github.com/arunm8489/YOLOv3-detection-from-scratch/blob/master/detected_messi.jpg)

![image](https://github.com/arunm8489/YOLOv3-detection-from-scratch/blob/master/dog-cycle-car.png)
![detected](https://github.com/arunm8489/YOLOv3-detection-from-scratch/blob/master/detected_dog-cycle-car.png)



