import os
import cv2
import numpy as np
import sys 
import tools.find_mxnet
import mxnet as mx
import importlib
from timeit import default_timer as timer
from detect.detector import Detector
import argparse
import threading
import time
import socket
import json
import jsonpickle

isFirstTimeSend = True
serverIp = '192.168.1.235'
serverPort = 5111
isNotQuit = True
numPeople =0 
numChair =0
batch = 1
data_shape = 416

class FactMessage(object):
	def __init__(self, target, commandType, factType, factProperty):
		self.target = target
		self.commandType = commandType
		self.factType = factType
		self.factProperty = factProperty
	def updateData(self, factType, factProperty):
		self.factType = factType
		self.factProperty = factProperty
	def jsonDefault(object):
		return object.__dict__

class dObject(object):
	def __init__(self, id, name, locatedAtRoomId, numOfSitting, numOfStanding, numOfLying):
		self.id = id
		self.name = name
		self.locatedAtRoomId = locatedAtRoomId
		self.numOfSitting = numOfSitting
		self.numOfStanding = numOfStanding
		self.numOfLying = numOfLying
	def updateData(self, numOfSitting, numOfStanding, numOfLying):
		self.numOfSitting = numOfSitting
		self.numOfStanding = numOfStanding
		self.numOfLying = numOfLying
	def jsonDefault(object):
		return object.__dict__

def socketConnect():
	while True:
		try:
			sSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			sSocket.settimeout(2)
			sSocket.connect((serverIp, serverPort))
			sys.stdout.flush()
			print "Socket: Connected"
            #return sSocket.makefile('w')
			return sSocket
		except socket.error as e:
			print("Socket: Error -> {0}".format(e))
			print "Socket: Reconnect"
			time.sleep(5)
import random
people = dObject('PYTXSTBPWKBSRJM84AR5FFEB', 'Camera1', 'room1', 0, 0, 0)
def sendFactPeriodly():	
	try:
		print "Socket: Start"
		global isFirstTimeSend	
		global isNotQuit	
		# Init Cosntant Device
		global people
		global numPeople
		
		factMessage = FactMessage('Fact', 'INSERT', 'Camera', people)
		message = ''
		count = 0
		
		# Create a TCP/IP socket
		sSocket = socketConnect()		
		while isNotQuit:								
			count = count + 1
			print ('Socket: %d' %(count))	
			sit = random.randint(0,4)
			st = random.randint(0,4-sit)
			ly = random.randint(0,4-sit-st)
			people.updateData(st,sit,ly)
			factMessage.updateData('Camera', people)					
         		#factMessage.updateData('people', people)								
							
			message = jsonpickle.encode(factMessage, unpicklable=False) + '\n'										
			print ('Socket {0}: Message -> {1}'.format(count, message.encode('utf8')))	
			try:
				sSocket.send(message.encode('utf8'))
				sys.stdout.flush()				
			except socket.error, exc:
				print "Socket: Error -> %s" % exc
				print "Socket: Reconnect"					
				sSocket = socketConnect()
			time.sleep(5)
			#if count>4:
			#	isNotQuit = False
			if isFirstTimeSend == True :
				print "Socket: First Time"
				isFirstTimeSend = False
			#time.sleep(10)
			#break
	except KeyboardInterrupt:
		print "Socket: Exit"
	except Exception as inst:
		print "Socket: Error -> ", inst
	finally:
		isNotQuit = False
		sSocket.close()
		exit()

#For detection
def get_batch(imgs):
	    img_len = len(imgs)
	    l = []
	    for i in range(batch):
		if i < img_len:
		    img = np.swapaxes(imgs[i], 0, 2)
		    img = np.swapaxes(img, 1, 2) 
		    img = img[np.newaxis, :] 
		    l.append(img[0])
		else:
		    l.append(np.zeros(shape=(3, data_shape, data_shape)))
	    l = np.array(l)
	    return [mx.nd.array(l)]

def peopleDetect():

	CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
	cap = cv2.VideoCapture(1)
	net =None
	prefix = os.path.join(os.getcwd(), 'model', 'yolo2_darknet19_416')
	epoch = 0
	
	mean_pixels = (123,117,104)
	ctx = mx.gpu(0)
	global numPeople
	global isNotQuit
	count = 0

	ret1, frame1 = cap.read()
	detector = Detector(net, prefix, epoch, data_shape, mean_pixels, ctx=ctx,batch_size = batch)
	while isNotQuit:
	    count+=1
	    ret, frame = cap.read()
	    ims = [cv2.resize(frame,(data_shape,data_shape)) for i in range(batch)]
	
	    data = None
	    data  = get_batch(ims)

	    start = timer()

	    det_batch = mx.io.DataBatch(data,[])
	    detector.mod.forward(det_batch, is_train=False)
	    detections = detector.mod.get_outputs()[0].asnumpy()
	    result = []
		    
	    for i in range(detections.shape[0]):
		det = detections[i, :, :]
		res = det[np.where(det[:, 0] >= 0)[0]]
		result.append(res)
	    time_elapsed = timer() - start
	   # print("Detection time for {} images: {:.4f} sec , fps : {:.4f}".format(batch*1, time_elapsed , (batch*1/time_elapsed)))
	    numPeople, numChair = detector.show_result(frame, det, CLASSES, 0.5,batch*1/time_elapsed )
	   # if count>40:
	#	isNotQuit = False
	    #break
	cap.release()
	cv2.destroyAllWindows()


#threading.Thread(target=peopleDetect).start()
threading.Thread(target=sendFactPeriodly).start()

