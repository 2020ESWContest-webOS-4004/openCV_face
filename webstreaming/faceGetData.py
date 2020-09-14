#**
#*  service         :   faceGetData.py
#*  type            :   python3
#*  date            :   2020.09.14
#*  author          :   한지훈(RORA)
#*  description     :   사용자 얼굴 메타 데이터 생성 프로그램
#**

# -*- coding: utf-8 -*- #
import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import pickle
import json
from PIL import Image


#얼굴 인식용 xml 파일 
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
#모델 생성 
model = cv2.face.LBPHFaceRecognizer_create()

class FaceGetData:
    def __init__(self):
        self.img_train_path = {}
        self.name_path = {}
        self.img_name_match = {}
        self.basic_file_path=[]
        self.train_file_path=[]
    #전체 사진에서 얼굴 부위만 잘라 리턴

    def face_extractor(self, img):
        #흑백처리 
        self.gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #얼굴 찾기 
        self.faces = face_classifier.detectMultiScale(self.gray, 1.3, 5)
        #찾은 얼굴이 없으면 None으로 리턴 
        if self.faces ==():
            return None
        #얼굴들이 있으면 
        for(x,y,w,h) in self.faces:
            self.cropped_face = img[y:y+h, x:x+w]
        #cropped_face 리턴 
        return self.cropped_face

    def createFolder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error : Creating directory. ' + directory)

    def saveTrainName(self, name, tpath):
        self.img_train_path[name] = tpath

    def saveName(self, name, path):
        self.name_path[name] = path

    def savePath(self, path):
        self.basic_file_path = './faces/' + path + '/' 
        return self.basic_file_path
    
    def saveImgName(self, name, count):
        self.img_name_match[name] = count

    def inputName(self):
        self.name = input('input your name : ')
        return self.name 

if __name__ =="__main__":
    a=FaceGetData()
    name = a.inputName()
    a.createFolder('./faces/'+name)
    #얼굴 사진 데이터 저장 폴더 경로
    a.basic_file_path = a.savePath(name)
    #카메라 실행 
    a.saveName(name, a.basic_file_path)

    cap = cv2.VideoCapture(0)
    #저장할 이미지 카운트 변수
    count = 0
    while True:
        #카메라로 부터 사진 1장 얻기 
        ret, frame = cap.read()
        #얼굴 감지 하여 얼굴만 가져오기 
        if a.face_extractor(frame) is not None:
            count+=1
            #얼굴 이미지 크기를 200x200으로 조정 
            face = cv2.resize(a.face_extractor(frame),(200,200))
            #조정된 이미지를 흑백으로 변환 
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            #파일명 User.(count).jpg로 저장
            file_name_path = a.basic_file_path +'User.'+ str(count)+'.jpg'          
            cv2.imwrite(file_name_path,face)
            
            #화면에 얼굴과 현재 저장 개수 표시          
            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow('Face Cropper',face)
        else:
            print("Face not Found")
            pass

        if cv2.waitKey(1)==13 or count==50:
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Colleting Samples Complete!!!')

    #faces폴더에 있는 파일 리스트 얻기
    data_path = a.basic_file_path
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
    #trainer/personal에 있는 파일 갯수 얻기
    train_path = 'trainer/personal'
    countfolder = [f for f in listdir(train_path) if isfile(join(train_path,f))]
    
    if len(countfolder) == 0:
        icount = 0
    else :
        icount = len(countfolder)

    #데이터와 매칭될 라벨 변수
    Training_Data, IDs= [], []

    #파일 개수만큼 루프
    for i, files in enumerate(onlyfiles):    
        image_path = data_path + onlyfiles[i]
        #이미지 불러오기
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        #이미지 파일이 아니거나 못 읽어 왔다면 무시
        if images is None:
            continue    
        #Training_Data 리스트에 이미지를 바이트 배열로 추가 
        Training_Data.append(np.array(images, dtype=np.uint8))
        #IDs 리스트엔 카운트 번호 추가 
        IDs.append(icount)
           
    #훈련할 데이터가 없다면 종료.
    if len(IDs) == 0:
        print("There is no data to train.")
        exit()

    #IDs를 32비트 정수로 변환
    IDs = np.asarray(IDs, dtype=np.int32)
    #학습 시작
    model.train(np.asarray(Training_Data), IDs)
    #개인 .yml 파일 생성
    model.write('trainer/personal/'+data_path.split('/')[2]+'.yml')
    #전체 .yml 파일 생성
    model.write('trainer/face_train.yml')
    tp='trainer/personal'+data_path.split('/')[2]+'.yml'
    a.saveTrainName(name, tp)
    a.saveImgName(name,icount)
    print("Model Training Complete!!!!!")

"""     import pymysql

    conn = pymysql.connect(host='', user='', password='', db='' ,charset='utf8') #DB 연결
    cur = conn.cursor(pymysql.cursors.DictCursor) #디폴트 커서 생성

    sql = "INSERT INTO bio_auth (jarvis_member_id,image_path,yml) VALUES (%s ,%s, %s);"
    val = (name, a.basic_file_path,'trainer/personal/' + data_path.split('/')[2]+'.yml')
    cur.execute(sql, val)
    
    conn.commit() """
    print('rowcount: ', cur.rowcount)

    conn.close() 

    #각 이미지 이름 및 경로에 대한 정보 파일로 저장
    with open('img_name_match.pickle','a+b') as fw:
        pickle.dump(a.img_name_match, fw)

    with open('img_train_path.pickle','a+b') as fw:
        pickle.dump(a.img_train_path, fw)

    with open('name_path.pickle','a+b') as fw:
        pickle.dump(a.name_path, fw)

    with open('name.json', 'r') as json_file:
        json_data = json.load(json_file)

        json_data['info'].append({
            'id' : icount,
            'name' : name
        })

    with open('name.json', 'w', encoding="utf-8") as outfile:
        json.dump(json_data, outfile, indent=4)
