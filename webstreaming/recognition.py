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
data_path = 'trainer/personal/'
with open('img_name_match.pickle', 'rb') as fr:
    nameList = []
    while True:
        try:
            data = pickle.load(fr)
        except EOFError:
            break        
        for i in data:
            nameList.append(i)

onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
print(onlyfiles)

for i in onlyfiles:
    print(i)
    model.read(data_path+i)
        

face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
ds_factor=0.6

class FaceRecognition:
    def __init__(self):
        #capturing video
        self.video = cv2.VideoCapture(0)
        self.found = []
        self.count1 = 0 
        self.count2 = 0
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
                    nameList.append(i)
        return nameList

    def face_found(self, face):
        if face != 0:
            self.found.append(face)
            if len(self.found) < 20:
                for i in self.found:
                    if i == 1:
                        self.count1 = self.count1 + 1
                    else :
                        self.count2 = self.count2 + 1 
                
                if self.count1 >= self.count2 :
                    self.face_result = 555
                    return self.face_result
                else :
                    self.face_result = 333
                    return self.face_result
            else :
                pass
        elif face == 0:
            self.found.append(face)
            if len(self.found) == 10:
                self.face_result = 111
                return self.face_result
        

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
        facefound = 0

        try:
            #detected pic transform to grayscale
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            #training model try predict
            id, result = model.predict(face)
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
                #test = self.face_found(facefound)
                cv2.putText(frame, label + " Unlocked", (0, 700), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                ret, jpeg = cv2.imencode('.jpg', frame)
                return jpeg.tobytes(), facefound
                
            else:
                #under 75 other person return Locked!!! 
                facefound = 2
                #test = self.face_found(facefound)
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
