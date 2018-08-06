from pose.estimator import TfPoseEstimator
from pose.networks import get_graph_path
from utils.sort import Sort
from utils.actions import actionPredictor
from utils.joint_preprocess import *
import sys
import cv2, os
import numpy as np
import time
import settings

poseEstimator = None

#fourcc = cv2.VideoWriter_fourcc(*'XVID')
# Get current height of frame

#out = cv2.VideoWriter('fishpose.avi', fourcc, 20.0, (640, 480))
def load_model():
    global poseEstimator
    poseEstimator = TfPoseEstimator(
        get_graph_path('mobilenet_thin'), target_size=(432, 368))


if __name__ == '__main__':
    load_model()
    print("Load all models done!")
    print("The system starts to run.")
    tracker = Sort(settings.sort_max_age, settings.sort_min_hit)

    #CAM_NUM = '/media/nvidia/9016-4EF8/pose/Dataset-maker-for-action-recognition/test1.mp4'
    CAM_NUM ='fishpose2.avi'
    CAM_NUM = 0
    cap = cv2.VideoCapture(CAM_NUM)
    fps = 0.00
    data = {}
    memory = {}
    joints = []
    current = []
    previous = []

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.winWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.winHeight)

    while True:
        start = time.time()
        ret, frame = cap.read()
	#print(frame.s
        #ret = True
        #frame = cv2.imread(filedir+file)
        show = cv2.resize(frame, (settings.winWidth, settings.winHeight))
	#out.write(show)
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)

        if ret:
            humans = poseEstimator.inference(show)
            ori = np.copy(show)
            show, joints, bboxes, xcenter, sk = TfPoseEstimator.get_skeleton(show, humans, imgcopy=False)
            height = show.shape[0]
            width = show.shape[1]
            if bboxes:
                result = np.array(bboxes)
                det = result[:, 0:5]
                det[:, 0] = det[:, 0] * width
                det[:, 1] = det[:, 1] * height
                det[:, 2] = det[:, 2] * width
                det[:, 3] = det[:, 3] * height
                trackers = tracker.update(det)
                current = [i[-1] for i in trackers]

                if len(previous) > 0:
                    for item in previous:
                        if item not in current and item in data:
                            del data[item]
                        if item not in current and item in memory:
                            del memory[item]

                previous = current
                for d in trackers:
                    xmin = int(d[0])
                    ymin = int(d[1])
                    xmax = int(d[2])
                    ymax = int(d[3])
                    label = int(d[4])
                    try:
                        j = np.argmin(np.array([abs(i - (xmax + xmin) / 2.) for i in xcenter]))
                    except:
                        j = 0
                    #print(j, label)
                    if joint_filter(joints[j]):
                        joints[j] = joint_completion(joint_completion(joints[j]))
                        if label not in data:
                            data[label] = [joints[j]]
                            memory[label] = 0
                        else:
                            data[label].append(joints[j])

                        if len(data[label]) == settings.L:
                            # print(data[label])
                            pred = actionPredictor().move_status(data[label])
                            if pred == 0:
                                pred = memory[label]
                            else:
                                memory[label] = pred
                            data[label].pop(0)

                            location = data[label][-1][1]
                            if location[0] <= 30:
                                location = (51, location[1])
                            if location[1] <= 10:
                                location = (location[0], 31)

                            #cv2.putText(sk, settings.move_status[pred], (location[0] - 30, location[1] - 10),
                                      #  cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                       # (0, 255, 0), 2)

                   # cv2.rectangle(sk, (xmin, ymin), (xmax, ymax),
                   #               (int(settings.c[label % 32, 0]),
                    #               int(settings.c[label % 32, 1]),
                   #                int(settings.c[label % 32, 2])), 4)
            #break
            end = time.time()
            fps = 1. / (end - start)
            cv2.putText(sk, 'FPS: %.2f' % fps, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            show = cv2.cvtColor(sk, cv2.COLOR_RGB2BGR)
            cv2.imshow("test", show)
            k = cv2.waitKey(5) & 0xFF
            if k == ord('q'):
                break
    #out.release()
    cap.release()
    cv2.destroyAllWindows()
