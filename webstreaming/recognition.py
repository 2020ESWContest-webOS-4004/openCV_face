#**
#*  service         :   recognition.py
#*  type            :   python3
#*  date            :   2020.10.03
#*  author          :   한지훈(RORA)
#*  description     :   사용자 얼굴 인식 및 판별 프로그램
#**
# -*- coding: utf-8 -*- #
import cv2
import numpy as np
import os
from os import listdir
from os.path import isdir, isfile, join
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

#얼굴 분류기 불러오기
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
ds_factor=0.6

class FaceRecognition:
    def __init__(self):
        #capturing video
        self.video = cv2.VideoCapture(0)
        self.nameList = []

    def __del__(self):
        #카메라 release
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
        
    #사용자 얼굴 학습
    def train(self, name):
        data_path = 'faces/' + name + '/'
        #파일만 리스트로 만듬
        face_pics = [f for f in listdir(data_path) if isfile(join(data_path,f))]
        Training_Data, Labels = [], []
        for i, files in enumerate(face_pics):
            image_path = data_path + face_pics[i]
            images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # 이미지가 아니면 패스
            if images is None:
                continue    
            Training_Data.append(np.asarray(images, dtype=np.uint8))
            Labels.append(i)
        if len(Labels) == 0:
            print("There is no data to train.")
            return None
        Labels = np.asarray(Labels, dtype=np.int32)
        # 모델 생성
        model = cv2.face.LBPHFaceRecognizer_create()
        # 학습
        model.train(np.asarray(Training_Data), np.asarray(Labels))
        #print(name + " : Model Training Complete!!!!!")

        #학습 모델 리턴
        return model

    def trains(self):        
        #faces 폴더의 하위 폴더를 학습
        data_path = './faces/'
        # 폴더만 색출
        model_dirs = [f for f in listdir(data_path) if isdir(join(data_path,f))]
        #학습 모델 저장할 딕셔너리
        models = {}
        # 각 폴더에 있는 얼굴들 학습
        for model in model_dirs:
            print('model :' + model)
            # 학습 시작
            result = self.train(model)
            # 학습이 안되었다면 패스!
            if result is None:
                continue
            # 학습되었으면 저장
            print('model2 :' + model)
            models[model] = result

        # 학습된 모델 딕셔너리 리턴
        return models    

    #얼굴 인식
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

    def trainer_get(self):
        data_path = 'trainer/personal/'
        onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
        return onlyfiles

    #실시간 영상
    def get_test(self):
        ret, frame = self.video.read()
        frame=cv2.flip(frame, 1)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def get_frame(self):
        #카메라부터 비디오 데이터 얻기
        ret, frame = self.video.read()
        frame=cv2.flip(frame, 1)
        nameDataList = []
        #얼굴 detect 시도
        frame,face = self.face_detector(frame)
        nameDataList = self.get_pickle('img_name_match.pickle')
        
        try:
            min_score = 999
            min_score_name = ""
            #발견된 얼굴을 그레이스케일로 변환
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            #트레이너 데이터 불러오기
            
            tmodels = self.trains()
            
            for key, model in tmodels.items():
                result = model.predict(face)
                
                if min_score > result[1]:
                    min_score = result[1]
                    min_score_name = key
            
            
            if min_score < 500:
                confidence = int(100*(1-(min_score)/300))
                # confidence 출력하기
                display_string = str(confidence)+'% Confidence'

            cv2.putText(frame,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(255,0,204),2)
            
            #75이상일 시, 사용자 이름과 Unlocked를 출력
            if confidence > 75:
                cv2.putText(frame, "Unlocked : " + min_score_name, (0, 700), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                ret, jpeg = cv2.imencode('.jpg', frame)
                #실시간 영상, 사용자 이름 리턴
                return jpeg.tobytes(), min_score_name
                
            else:
                min_score_name = '' 
                #75 이하일때, Locked을 출력
                cv2.putText(frame, "Locked", (0, 700), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                ret, jpeg = cv2.imencode('.jpg', frame)
                return jpeg.tobytes(), min_score_name
                
        except:
            min_score_name = ''
            #얼굴을 못찾을 시 출력
            cv2.putText(frame, "Face Not Found", (0, 700), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 10, 10), 2)
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes(), min_score_name
            pass
