import cv2
import numpy as np
import time
from  queue import Queue
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
import torch

from PIL import Image
from thread.v5_modify.detect_yolov5 import Detection, Tracking
from thread.v5_modify.predict_yolo import Predict
from thread.v5_modify.tracker import *
from thread.Capture import CaptureThread
import sys, os
import cvzone


class Inference(QThread):
    signal = pyqtSignal(np.ndarray)
    def __init__(self, queue):
        super().__init__()
        self.frame = None
        self.threadActive = False
        self.queue_cap = queue
        self.device = None
        self.out_file = None
        self.classes = None
        self.model = None
        self.tracker = Tracker()
        self.clf_brand = Predict()
        self.clf_color = Predict()
        self.detection= Detection()
        self.detection.weights = 'model/yolov5s.engine'
        
        self.clf_brand.weights = 'model/make.engine'

        self.clf_color.weights = 'model/color.pt'
        self.threadActive = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("CUDA device name:", torch.cuda.get_device_name(self.device))

        self.detection._load_model()
        self.clf_brand._load_model()
        self.clf_color._load_model()
        
        self.fpsReader = cvzone.FPS()
        # print("thread inference start")


    def run(self):
        print("thread inference start")
        while self.threadActive:
            
            if self.queue_cap.qsize() > 0:
                

                frame = self.queue_cap.get()   
                result11 = self.detection.detect(frame)
                for box in result11:
                    x1 ,y1, x2, y2 , cls, conf = box
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (0,0,255),2)
                    img_car = frame[y1:y2, x1:x2]
                    model = self.clf_brand.predict(img_car)
                    color = self.clf_color.predict(img_car)
                    cv2.putText(frame, str(model) + '/' + str(color) ,(x2,y2), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)
                    
                fps, frame = self.fpsReader.update(frame, pos=(100, 100), color=(255, 0, 0), scale=3, thickness=3)
                # cv2.putText(frame, str(round(fps)),(100,100), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)
                self.signal.emit(frame)
                time.sleep(0.001)
            time.sleep(0.001)
    


    
