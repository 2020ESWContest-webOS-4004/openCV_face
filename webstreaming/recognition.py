#**
#*  service         :   recognition.py
#*  type            :   python3
#*  date            :   2020.09.14
#*  author          :   한지훈(RORA)
#*  description     :   사용자 얼굴 인식 및 판별 프로그램
#**

# -*- coding: utf-8 -*- #
import re
import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import glob
import pickle
import json

model = cv2.face.LBPHFaceRecognizer_create()
train_path = 'trainer/personal'

countfolder = [f for f in listdir(train_path) if isfile(join(train_path,f))]
if len(countfolder) == 0:
    icount = 0
else :
    icount = len(countfolder)

""" data_path = 'trainer/personal/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

for i in onlyfiles:
    model.read(data_path+i) """

data_path = 'trainer/face_train.yml'
model.read(data_path)

face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
ds_factor=0.6

class FaceRecognition:
    def __init__(self):
        #capturing video
        self.video = cv2.VideoCapture(0)
        self.found = []
        self.nameList = []
        self.face_result = 0

    def __del__(self):
        #releasing camera
        self.video.release()

    def get_pickle(self, fname):
        with open(fname, 'rb') as fr:
            while True:
                try:
                    data = pickle.load(fr)
                except EOFError:
                    break
                for i in data:
                    self.nameList.append(i)           
        return self.nameList

    def face_detector(self, frame):
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects=face_classifier.detectMultiScale(gray,1.3,5)

        if face_rects == ():
            return frame, [] 

        for(x,y,w,h) in face_rects:
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,255),2)
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (200,200))
            break
        return frame, face

    def get_frame(self):
        # get video data from camera 
        ret, frame = self.video.read()
        frame=cv2.flip(frame, 1)
        nameDataList = []
        # try face detect
        frame,face = self.face_detector(frame)
        nameDataList = self.get_pickle('img_name_match.pickle')
        
        try:
            #detected pic transform to grayscale
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            #training model try predict
            id, result = model.predict(face)
            #result is confidence and close to 0
            #this mean registered user
            label = nameDataList[id]
            
            if result < 500:
                confidence = int(100*(1-(result)/300))
                # display confidence
                display_string = str(confidence)+'% Confidence'

            cv2.putText(frame,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(255,0,204),2)
            
            #over 75 same persone return UnLocked! 
            if confidence > 75:
                cv2.putText(frame, label + " Unlocked", (0, 700), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                ret, jpeg = cv2.imencode('.jpg', frame)
                return jpeg.tobytes(), label
                
            else:
                label = '' 
                #under 75 other person return Locked!!! 
                cv2.putText(frame, "Locked", (0, 700), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                ret, jpeg = cv2.imencode('.jpg', frame)
                return jpeg.tobytes(), label
                
        except:
            label = ''
            #not face recognition 
            cv2.putText(frame, "Face Not Found", (0, 700), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 10, 10), 2)
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes(), label
            pass
