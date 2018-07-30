import os
import cv2
import numpy as np
import sys 
import src.find_mxnet
import mxnet as mx
import importlib
prefix = os.getcwd()
import imutils
sys.path.append(prefix + '/kcf')
import KCF
from timeit import default_timer as timer
from src.detector import Detector
import argparse
import threading
import time
import socket
import json
import jsonpickle
from src.zigbee import Zigbee
from src.mms import MMS 
isFirstTimeSend = True
serverIp = '192.168.1.235'
serverPort = 5111
isNotQuit = True
numOfStanding =0 
numOfSitting =0
numOfLying = 0

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
		global numOfStanding
    		global numOfSitting
    		global numOfLying
		
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
			people.updateData(numOfSitting,numOfStanding,numOfLying)
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
		#sSocket.close()
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


class Object:
    def __init__(self, xmin1, ymin1, xmax1, ymax1, label1, score, newlabel):
        self.xmin = xmin1
        self.ymin = ymin1
        self.xmax = xmax1
        self.ymax = ymax1
        self.label = label1
	self.score = score
	self.newlabel = newlabel
CLASSES = ('standing', 'sitting', 'lying')

def parse_args():
    parser = argparse.ArgumentParser(description='Single-shot detection network demo')
    parser.add_argument('--network', dest='network', type=str, default='yolo2_test_416',
                        help='which network to use')
    parser.add_argument('--video', dest='video', type=str, default='/dev/video0',
                        help='run demo with images, use comma(without space) to seperate multiple images')
    parser.add_argument('--dir', dest='dir', nargs='?',
                        help='demo image directory, optional', type=str)
    parser.add_argument('--ext', dest='extension', help='image extension, optional',
                        type=str, nargs='?')
    parser.add_argument('--epoch', dest='epoch', help='epoch of trained model',
                        default=500, type=int)
    parser.add_argument('--prefix', dest='prefix', help='trained model prefix',
                        default=os.path.join(os.getcwd(), 'model/'), type=str)
    parser.add_argument('--cpu', dest='cpu', help='(override GPU) use CPU to detect',
                        action='store_true', default=False)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0,
                        help='GPU device id to detect with')
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=416,
                        help='set image shape')
    parser.add_argument('--mean-r', dest='mean_r', type=float, default=123,
                        help='red mean value')
    parser.add_argument('--mean-g', dest='mean_g', type=float, default=117,
                        help='green mean value')
    parser.add_argument('--mean-b', dest='mean_b', type=float, default=104,
                        help='blue mean value')
    parser.add_argument('--thresh', dest='thresh', type=float, default=0.3,
                        help='object visualize score threshold, default 0.6')
    parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.9,
                        help='non-maximum suppression threshold, default 0.5')
    parser.add_argument('--force', dest='force_nms', type=bool, default=True,
                        help='force non-maximum suppression on different class')
    parser.add_argument('--timer', dest='show_timer', type=bool, default=True,
                        help='show detection time')
    parser.add_argument('--deploy', dest='deploy_net', action='store_true', default=False,
                        help='Load network from json file, rather than from symbol')
    args = parser.parse_args()
    return args
def iou_fiter(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = (xB - xA + 1) * (yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 	minarea = min(boxAArea, boxBArea)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(minarea)
 
	# return the intersection over union value
	if xB>xA and yB>yA: 
	    return iou
	else:
	    return 0
def iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# return the intersection over union value
	if xB>xA and yB>yA: 
	    interArea = (xB - xA + 1) * (yB - yA + 1)
	    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	    iou = interArea / float(boxAArea+boxBArea-interArea)
	    return iou
	else:
	    return 0
def behaviourDetect():
    zig = Zigbee()
    mms_count = 100
    detect_lying = False
    mms = MMS()
    args = parse_args()
    if args.cpu:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu_id)

    # parse image list
    image_list = ['messigray.png']
    assert len(image_list) > 0, "No valid image specified to detect"
    prefix = args.prefix + args.network
    network = None
    detector = Detector(network, prefix, args.epoch, args.data_shape, (args.mean_r, args.mean_g, args.mean_b), ctx=ctx)
    # run detection
    global isNotQuit
    global numOfStanding
    global numOfSitting
    global numOfLying
    a = True
    kcf = False
    cap = cv2.VideoCapture(args.video)
    #cap = cv2.VideoCapture(0)
    objects = []
    pre_objects = []
    fobjects = []
    ret, frame = cap.read()
    cap.set(3,1920)
    cap.set(4,1080)
    frame = cv2.flip( frame, 1 )
    angle = 30
    frame = imutils.rotate(frame, angle)
    frame = cv2.flip( frame, 0 )

    height,width = frame.shape[:2]
    frame = cv2.resize(frame,(width/2,height/2))
    pre_Frame = frame
    cv2.imwrite('messigray.png',frame)
    test_iter = detector.im_detect(image_list, args.dir, args.extension, show_timer=args.show_timer)
    font = cv2.FONT_HERSHEY_SIMPLEX
    while(isNotQuit):
	detect_lying = False
	if(mms_count < 100):
		mms_count += 1
	start = timer()
#KCF track
	if(kcf):
	    ret1, framekcf = cap.read()
	    framekcf = cv2.flip( framekcf, 1 )
	    angle = 30
    	    framekcf = imutils.rotate(framekcf, angle)
	    framekcf = cv2.flip( framekcf, 0 )
	    #cap.set(3,1920)
	    #cap.set(4,1080)

            if not ret1:
			break
            height,width = framekcf.shape[:2]
            
	    framekcf = cv2.resize(framekcf,(width/2,height/2))
	    st= 0
	    sit = 0
	    ly = 0
	    for objecta in (objects): 
	        tracker = KCF.kcftracker(False, True, False, False) #hog, fixed_window, multiscale, lab
		tracker.init([objecta.xmin,objecta.ymin,objecta.xmax-objecta.xmin,objecta.ymax-objecta.ymin], pre_Frame)
		pre_Frame = framekcf
		boundingbox = tracker.update(framekcf)
		boundingbox = map(int, boundingbox)
		
		if objecta.label == 0:
			cv2.rectangle(framekcf,(boundingbox[0],boundingbox[1]), (boundingbox[0]+boundingbox[2],boundingbox[1]+boundingbox[3]), (0,0,0), 1)
			cv2.putText(framekcf,CLASSES[objecta.label],(boundingbox[0],boundingbox[1]), font, 0.3,(0,0,0),1,cv2.LINE_AA)
			st +=1
		if objecta.label == 1:
			cv2.rectangle(framekcf,(boundingbox[0],boundingbox[1]), (boundingbox[0]+boundingbox[2],boundingbox[1]+boundingbox[3]), (0,255,0), 1)
			cv2.putText(framekcf,CLASSES[objecta.label],(boundingbox[0],boundingbox[1]), font, 0.3,(0,255,0),1,cv2.LINE_AA)
			sit +=1
		if objecta.label == 2:
			cv2.rectangle(framekcf,(boundingbox[0],boundingbox[1]), (boundingbox[0]+boundingbox[2],boundingbox[1]+boundingbox[3]), (0,0,255), 1)
			cv2.putText(framekcf,CLASSES[objecta.label],(boundingbox[0],boundingbox[1]), font, 0.3,(0,0,255),1,cv2.LINE_AA)
			ly +=1
		#cv2.putText(framekcf,CLASSES[objecta.label]+str(objecta.score),(boundingbox[0],boundingbox[1]), font, 0.3,(255,255,255),1,cv2.LINE_AA)
	    framekcf = cv2.resize(framekcf,(width,height))
	    numOfStanding = st
    	    numOfSitting = sit
    	    numOfLying = ly
#~KCF track

	#pre process next frame
	    ret, frame = cap.read()
	    frame = cv2.flip( frame, 1 )
	    angle = 30
	    frame = imutils.rotate(frame, angle)
	    frame = cv2.flip( frame, 0 )
	    #cap.set(3,1920)
	    #cap.set(4,1080)

	    if not ret:
			break
	    frame = cv2.resize(frame,(width/2,height/2))
	    pre_Frame = frame
	    cv2.imwrite('messigray.png',frame)
       	    test_iter = detector.im_detect(image_list, args.dir, args.extension, show_timer=args.show_timer)
	    cv2.imshow("img",framekcf)
	    kcf = False
	    pre_objects = objects
	    objects = []
	#~pre process next frame

	else:
#detection every 2 frame
            dets = detector.detect(test_iter, args.show_timer)
	    
#visualize detection
	    for k, det in enumerate(dets):
	        height = frame.shape[0]
                width = frame.shape[1]
                for i in range(det.shape[0]):
                    cls_id = int(det[i, 0])
                    if cls_id >= 0:
                        score = det[i, 1]
                        if score > args.thresh:
                            xmin = int(det[i, 2] * width)
                            ymin = int(det[i, 3] * height)
                            xmax = int(det[i, 4] * width)
                            ymax = int(det[i, 5] * height)
                        
                            #cv2.rectangle(frame,(xmin,ymin),(xmax, ymax),(0,255,255),1)

                            class_name = str(cls_id)
                            if CLASSES and len(CLASSES) > cls_id:
                                class_name = CLASSES[cls_id]
			    zig.count_detec(cls_id)
			    objecta= Object(xmin, ymin, xmax, ymax, cls_id, score, 1)
			    objects.append(objecta)
	    fobjects = []
	    t = threading.Thread(target=zig.send_zigbee, args=())
	    t.start()
	#filter object overlap

	    for aa in range(len(objects)):
		for bb in range(aa+1, len(objects)):
		    iou1 = iou_fiter([objects[aa].xmin,objects[aa].ymin,objects[aa].xmax,objects[aa].ymax],[objects[bb].xmin,objects[bb].ymin,objects[bb].xmax,objects[bb].ymax])
		    if iou1>0.6 and iou1<=1:
			if objects[aa].score > objects[bb].score:
			    fobjects.append(objects[bb])
			else:
			    fobjects.append(objects[aa])
	    for objecta in (fobjects):
		try : 
	            objects.remove(objecta)
		except :
		    print ' '
	#~filter object overlap

	    #correct object label
	    for aa in range(len(objects)):
		for bb in range(len(pre_objects)):
		    iou1 = iou([objects[aa].xmin,objects[aa].ymin,objects[aa].xmax,objects[aa].ymax],[pre_objects[bb].xmin,pre_objects[bb].ymin,pre_objects[bb].xmax,pre_objects[bb].ymax])
		    if iou1>0.6 and iou1<=1 and objects[aa].label != pre_objects[bb].label :
			objects[aa].newlabel = pre_objects[bb].newlabel +1
			if objects[aa].newlabel<=14 :
			    objects[aa].label = pre_objects[bb].label
			else :
			    objects[aa].newlabel =1
			    
	#~correct object label
	    st= 0
	    sit = 0
	    ly = 0
	    for objecta in (objects):
		#cv2.rectangle(frame,(objecta.xmin,objecta.ymin),(objecta.xmax, objecta.ymax),(0,255,0),1)
		if objecta.label == 0:
			cv2.rectangle(frame,(objecta.xmin,objecta.ymin),(objecta.xmax, objecta.ymax),(0,0,0),1)
			cv2.putText(frame,CLASSES[objecta.label],(objecta.xmin,objecta.ymin), font, 0.3,(0,0,0),1,cv2.LINE_AA)
			st +=1
		if objecta.label == 1:
			cv2.putText(frame,CLASSES[objecta.label],(objecta.xmin,objecta.ymin), font, 0.3,(0,255,0),1,cv2.LINE_AA)
			cv2.rectangle(frame,(objecta.xmin,objecta.ymin),(objecta.xmax, objecta.ymax),(0,255,0),1)
			sit +=1
		if objecta.label == 2:
			cv2.rectangle(frame,(objecta.xmin,objecta.ymin),(objecta.xmax, objecta.ymax),(0,0,255),1)
			cv2.putText(frame,CLASSES[objecta.label],(objecta.xmin,objecta.ymin), font, 0.3,(0,0,255),1,cv2.LINE_AA)
			detect_lying = True
			ly +=1
		#cv2.putText(frame,CLASSES[objecta.label]+str(objecta.score),(objecta.xmin,objecta.ymin), font, 0.3,(255,255,255),1,cv2.LINE_AA)
#~visualize detection

	    frame = cv2.resize(frame,(width*2,height*2)) #resize frame
            cv2.imshow("img",frame)  #show video
	    numOfStanding = st
    	    numOfSitting = sit
    	    numOfLying = ly
	    if(detect_lying and mms_count == 100):
		mms_count = 0
		frame_mms = cv2.resize(frame,(420,320))
		cv2.imwrite('mms_save.png',frame_mms)
		tt = threading.Thread(target=mms.send_mms, args=())
		tt.start()
	    kcf = True
#~detection 

        time_elapsed = timer() - start
        #print("Detection timessssss for {} images: {:.4f} sec fps {:.4f}".format(
        #        1, time_elapsed, 1/time_elapsed))
	k = cv2.waitKey(1)& 0xFF
	if k == ord('q'):
		isNotQuit = False
		break
    cap.release()
    cv2.destroyAllWindows()


threading.Thread(target=behaviourDetect).start()
threading.Thread(target=sendFactPeriodly).start()
