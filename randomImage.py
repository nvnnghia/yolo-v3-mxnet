import os
from random import shuffle
import cv2
import shutil
'''
	get 500 random images form VOC2007 test dataset.
'''

with open('data/VOCdevkit/VOC2007/test1.txt', 'r') as f:
        lines = f.readlines()
imagenames = [x.strip() for x in lines]
shuffle(imagenames)
imagenames500 =  imagenames[0:500]
image_list = ['data/VOCdevkit/VOC2007/JPEGImages/'+name + '.jpg' for name in imagenames500]
image_save_list = ['output/RandomImage/'+name + '.jpg' for name in imagenames500]
#print image_list
if os.path.exists("output/RandomImage"):
	shutil.rmtree('output/RandomImage')
os.makedirs("output/RandomImage")
for i, name in enumerate(image_list):
	image = cv2.imread(name)
	cv2.imwrite(os.path.join(image_save_list[i]),image)
