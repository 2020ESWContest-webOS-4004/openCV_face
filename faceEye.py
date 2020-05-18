import numpy as np
import cv2 as cv

# 캠에서 영상을 읽어옴
cap = cv.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

# 얼굴과 눈을 검출하기 위해 미리 학습시켜 놓은 XML 포맷으로 저장된 분류기를 로드
face_cascade = cv.CascadeClassifier('./data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('./data/haarcascades/haarcascade_eye.xml')


while(True):

    ret, frame = cap.read()
    # 얼굴과 눈을 검출할 그레이스케일 이미지를 준비
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 이미지에서 얼굴을 검출
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # 얼굴이 검출되었다면 얼굴 위치에 대한 좌표 정보를 리턴
    for (x,y,w,h) in faces:

        # 원본 이미지에 얼굴의 위치를 표시
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        # 눈 검출은 얼굴이 검출된 영역 내부에서만 진행하기 위해 ROI를 생성
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # 눈을 검출
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # 눈이 검출되었다면 눈 위치에 대한 좌표 정보를 리턴
        for (ex,ey,ew,eh) in eyes:

            # 원본 이미지에 얼굴의 위치를 표시합니다. ROI에 표시하면 원본 이미지에도 표시
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()