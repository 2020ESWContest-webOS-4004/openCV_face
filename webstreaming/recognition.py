#**
#*  service         :   recognition.py
#*  type            :   python3
#*  date            :   2020.09.06
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

datalist=[]
ymlList=[]
descs = {}
model = cv2.face.LBPHFaceRecognizer_create()
train_path = 'trainer/personal'

countfolder = [f for f in listdir(train_path) if isfile(join(train_path,f))]
if len(countfolder) == 0:
    icount = 0
else :
    icount = len(countfolder)

data_path = 'trainer/personal/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

for i in onlyfiles:
    model.read(data_path+i)

face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
ds_factor=0.6

class FaceRecognition:
    def __init__(self):
        #capturing video
        self.video = cv2.VideoCapture(0)
        self.found = []
        self.a = 0 
        self.b = 0
        self.c = 0 
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
        
    def face_found(self, face):
        self.found.append(face)
        for i in self.found:
            if i == 1:
                self.a += 1
            elif i == 2 :
                self.b += 1 

        if len(self.found) == 40:
            if self.a >= self.b :
                self.face_result = 555
            else:
                self.face_result = 333

        if self.face_result != None:
            return self.face_result

    def get_frame(self):
        # get video data from camera 
        ret, frame = self.video.read()
        frame=cv2.flip(frame, 1)
        nameDataList = []
        # try face detect
        frame,face = self.face_detector(frame)
        nameDataList = self.get_pickle('img_name_match.pickle')
        facefound = 0

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
                facefound = 1
                cv2.putText(frame, label + " Unlocked", (0, 700), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                ret, jpeg = cv2.imencode('.jpg', frame)
                return jpeg.tobytes(), facefound
                
            else:
                #under 75 other person return Locked!!! 
                facefound = 2
                cv2.putText(frame, "Locked", (0, 700), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                ret, jpeg = cv2.imencode('.jpg', frame)
                return jpeg.tobytes(), facefound
                
        except:
            #not face recognition 
            facefound = 0
            cv2.putText(frame, "Face Not Found", (0, 700), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 10, 10), 2)
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes(), facefound
            pass
