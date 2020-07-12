import dlib
from darkflow.net.build import TFNet
import cv2
import time
import numpy as np
import math

from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils

import time

if __name__=="__main__":
    start = time.time()  # 시작 시간 저장
    options = {"model": "cfg/yolov2-voc.cfg", "load": "weights/yolov2-voc.weights", "threshold": 0.1}
    #options = {"model": "cfg/yolo.cfg", "load": "weights/yolo.weights", "threshold": 0.5}
    # YOLO 1 과 YOLO2 밖에 작동안함
    cap = cv2.VideoCapture("0611.mp4")
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print( length )
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (int(width), int(height)))
    tfnet = TFNet(options)
    ind = 1
    while(cap.isOpened()):
        ret , frame = cap.read()
        if ret == False:
            break
        if ind % 20 == 0:
            result = tfnet.return_predict(frame)
            print("Frame: ",ind) 
            for i in range (0,len(result)):
                if result[i]['label'] == 'person' : 
                    pt1 = (result[i]['topleft']['x'],result[i]['topleft']['y'])
                    pt2 = (result[i]['bottomright']['x'],result[i]['bottomright']['y'])
                    print(pt1,pt2)
                    cv2.rectangle(frame, pt1, pt2, (255, 255, 255), 1)    
            out.write(frame)
        ind += 1   
        if cv2.waitKey(1) == 27:
            break
    print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
    cap.release()
    out.release()
    cv2.destroyAllWindows()


