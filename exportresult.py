import argparse
import src.find_mxnet
import mxnet as mx
import os, cv2
import importlib
import sys
from src.detector import Detector
MXNET_CUDNN_AUTOTUNE_DEFAULT = 0
from random import shuffle
import shutil
import yaml

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
   
if __name__ == '__main__':
    
    ## LOAD CONFIG PARAMS ##
    if (os.path.isfile('config.yml')):
        with open("config.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
    else:
        with open("config.sample.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile)


    out_dir 		= cfg['out_dir']
    gpu 		= cfg['gpu']
    thresh 		= cfg['confidence_thresh']
    prefix 		= cfg['prefix']
    epoch		= cfg['epoch']
    data_shape		= cfg['data_shape']
    mean_r		= cfg['mean_r']
    mean_g		= cfg['mean_g']
    mean_b		= cfg['mean_b']
    list_images_name 	= cfg['list_images_name']
    image_folder 	= cfg['image_folder']
    out_image_folder 	= cfg['out_image_folder']
    ##~ LOAD CONFIG PARAMS ##

    if gpu:
        ctx = mx.gpu()
    else:
        ctx = mx.cpu()

    image_dir = os.path.join(image_folder)
    image_list=[]
    
    ## WRITE IMAGE's NAMES TO FILE ##
    file1 = open(out_dir+'/'+list_images_name,'w')
    for f in os.listdir(image_dir):
        image_list.append(os.path.join(image_dir, f))
	file1.write(f.split('.')[0]+'\n')
    file1.close()
    ##~ WRITE IMAGE's NAMES TO FILE ##

    filePerson = open(out_dir+'/out_person.txt','w')
    fileBicycle = open(out_dir+'/out_bicycle.txt','w')
    fileDog = open(out_dir+'/out_dog.txt','w')
    fileCat = open(out_dir+'/out_cat.txt','w')
    fileCar = open(out_dir+'/out_car.txt','w')

    network = None
    detector = Detector(network, prefix, epoch, data_shape, (mean_r, mean_g, mean_b), ctx=ctx)
    dets = detector.im_detect(image_list, show_timer=True)

    assert len(dets) == len(image_list)
    if os.path.exists(out_dir+'/'+out_image_folder):
	shutil.rmtree(out_dir+'/'+out_image_folder)
    os.makedirs(out_dir+'/'+out_image_folder)
    for k, det in enumerate(dets):
	img = cv2.imread(image_list[k])
	height = img.shape[0]
        width = img.shape[1]
	for i in range(det.shape[0]):
            cls_id = int(det[i, 0])
            if cls_id in [1,6,7,11,14]:
                score = det[i, 1]
                if score > thresh:
                    xmin = int(det[i, 2] * width)
                    ymin = int(det[i, 3] * height)
                    xmax = int(det[i, 4] * width)
                    ymax = int(det[i, 5] * height)
		    
                    if CLASSES and len(CLASSES) > cls_id:
                        class_name = CLASSES[cls_id]
		    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,255),2)
		    cv2.putText(img,class_name,(xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
		    if class_name=='bicycle':
			fileBicycle.write('%s %.2f %.2f %.2f %.2f %.2f\n' % ((image_list[k].split('/')[-1]).split('.')[0], score, xmin, ymin, xmax, ymax))
		    if class_name=='person':
			filePerson.write('%s %.2f %.2f %.2f %.2f %.2f\n' % ((image_list[k].split('/')[-1]).split('.')[0], score, xmin, ymin, xmax, ymax))
		    if class_name=='dog':
			fileDog.write('%s %.2f %.2f %.2f %.2f %.2f\n' % ((image_list[k].split('/')[-1]).split('.')[0], score, xmin, ymin, xmax, ymax))
		    if class_name=='car':
			fileCar.write('%s %.2f %.2f %.2f %.2f %.2f\n' % ((image_list[k].split('/')[-1]).split('.')[0], score, xmin, ymin, xmax, ymax))
		    if class_name=='cat':
			fileCat.write('%s %.2f %.2f %.2f %.2f %.2f\n' % ((image_list[k].split('/')[-1]).split('.')[0], score, xmin, ymin, xmax, ymax))

	cv2.imwrite(os.path.join(out_dir+'/'+out_image_folder+'/'+image_list[k].split('/')[-1]),img)
    filePerson.close()
    fileCat.close()
    fileCar.close()
    fileBicycle.close()
    fileDog.close()

