import numpy as numpy
from cv2 import cv2
# import cv2 as cv2
import torch
from utils import *
import argparse
import os
from model import Darknet


print(torch.__version__)
def argparser():
    """
    Parse arguements to the detect module
    
    """    
    parser = argparse.ArgumentParser(description='YOLOv3 Detection')   
    parser.add_argument("--input_image", dest = 'image', help = "image to perform detection")
    parser.add_argument("--weights",dest = 'weightsfile', help="yolo3 weights file",
                                    default = "yolov3.weights", type = str)
    parser.add_argument("--classes",dest='classfile',help="file containing classes", default='coco.names')

    return parser.parse_args()



args = argparser()
CUDA = False
cfgfile = os.path.join('cfg','yolov3.cfg')
weightfile = args.weightsfile
classfile = args.classfile

input_image = args.image
output_image_name  = 'detected_' + args.image
print(output_image_name)

nms_thesh = 0.5
#Set up the neural network
print("Loading network.....")
model = Darknet(cfgfile)
model.load_weights(weightfile)
print("Network successfully loaded")
classes = load_classes(classfile)
print('Classes loaded')



conf_inp_dim = int(model.net_info["height"])#608

# treading and resizing image
processed_image, original_image, original_img_dim = preprocess_image(input_image,conf_inp_dim)

im_dim = original_img_dim[0], original_img_dim[1]
im_dim = torch.FloatTensor(im_dim).repeat(1,2)

#If there's a GPU availible, put the model on GPU
if CUDA:
    im_dim = im_dim.cuda()
    model.cuda()

#Set the model in evaluation mode
model.eval()
with torch.no_grad():
      prediction = model(processed_image)


output = final_detection(prediction, confidence_threshold=0.5, num_classes=80, nms_conf = nms_thesh)



im_dim_list = torch.index_select(im_dim, 0, output[:,0].long())

scaling_factor = torch.min(conf_inp_dim/im_dim_list,1)[0].view(-1,1)
output[:,[1,3]] -= (conf_inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
output[:,[2,4]] -= (conf_inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
output[:,1:5] /= scaling_factor
    


# adjusting bounding box size between 0 and configuration image size
output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(conf_inp_dim))


list(map(lambda x: draw_boxes(x, original_image,classes), output))
cv2.imwrite(output_image_name, original_image)
print('Finished Prediction')