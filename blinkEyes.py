# -*- coding: utf-8 -*- #
from scipy.spatial import distance as dist
import numpy as np
import dlib
import cv2
import time, datetime

# 얼굴 랜드마크 0~68까지 마크업
JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))

# 눈 사이의 거리를 유클리드 거리로 계산 
def eye_aspect_ratio(eye):
		
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	C = dist.euclidean(eye[0], eye[3])

	ear = (A + B) / (2.0 * C)

	return ear

# 눈을 깜박이고 있다는 것을 알기 위한 임계값
EYE_AR_THRESH = 0.22
# 3개의 프레임이 연속으로 발생해야 깜박임이 등록된다는 것을 나타내기 위함
EYE_AR_CONSEC_FRAMES = 3
EAR_AVG = 0

# 깜박임 수 카운트
COUNTER = 0
TOTAL = 0

# 얼굴과 눈을 검출하기 위해 미리 학습시켜 놓은 XML 포맷으로 저장된 분류기를 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./data/shape_predictor_68_face_landmarks.dat')

while True:
    try:
        # 캠에서 영상을 읽어옴
        cap = cv2.VideoCapture(0)
    
    except:
        print("Camera Loading Error")

    ret, frame = cap.read()

    if not ret:
        break
    
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray,0)
        
        for rect in rects:
            x = rect.left()
            y = rect.top()
            x1 = rect.right()
            y1 = rect.bottom()

            landmarks = np.matrix([[p.x , p.y] for p in predictor(frame, rect).parts()])

            # 왼쪽 눈의 랜드마크 범위 설정
            left_eye = landmarks[LEFT_EYE_POINTS]
            # 오른쪽 눈의 랜드마크 범위 설정
            right_eye = landmarks[RIGHT_EYE_POINTS]

            # 눈 부위를 위한 랜드마크를 시각화하는 핸들러 구현
            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)


            ear_left = eye_aspect_ratio(left_eye)
            ear_right = eye_aspect_ratio(right_eye)
            ear_avg = (ear_left + ear_right) / 2.0
            
            # 임계값 이하로 눈꺼플의 위치가 변경되면 눈 감은것이라 인식
            if ear_avg < EYE_AR_THRESH:
                alarm = time.time()

                # 눈 감고 있는 시간이 1초가 지나면
                # Wake up Man이라는 문구를 출력 (나중엔 알람으로 변경 예정)
                if alarm > 1 :
                    COUNTER += 1
                    if COUNTER >= 5 :
                        print("Wake up Man")

            else :
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                    print("eye blinked")
                COUNTER = 0

            
            cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear_avg), (200, 30) ,cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #cv2.putText(frame, "TIME: ".format(times[0]), (500, 30) ,cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()