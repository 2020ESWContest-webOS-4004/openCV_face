#**
#*  service         :   main.py
#*  type            :   python3
#*  date            :   2020.10.03
#*  author          :   한지훈(RORA)
#*  description     :   플라스크 서버
#**

# -*- coding: utf-8 -*- #
from flask import Flask, render_template, Response,request
from flask_socketio import SocketIO, emit, send
import socketio as SocketIO_client
from recognition import FaceRecognition
import json
import os

#newCode="sudo modprobe bcm2835-v4l2"
#os.system(newCode)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socket_server = SocketIO(app)
socket_client = SocketIO_client.Client()

host = '0.0.0.0'
port = '5000'
count = 0 
face_id = 0 
jarvis_mi = ' '

# 클라이언트 연결
#socket_client.connect('http://192.168.1.8:5555')

@app.route('/')
def index():
    # 웹페이지 렌더링
    return render_template('index.html')

""" def get_json(l):
    global face_id
    with open('name.json', 'r') as json_file:
        json_data = json.load(json_file)
        i = 0 
        while True:
            try:
                j = json_data['info'][i]
                i = i +1
                if j['name'] == l:
                    face_id = j['id']
            except IndexError:
                break
    return face_id """

#DB 데이터 접속 및 사용자 조회
def sqlResult(name):
    import pymysql
    conn = pymysql.connect(host='', user='', password='', db='' ) #DB 연결
    cur = conn.cursor()
    sql = "select member_name from jarvis_member where jarvis_id ='{}'".format(name)
    cur.execute(sql)
    row = cur.fetchone()
    while row:
        name = row[0]
        row = cur.fetchone()

    if name is not None:
        conn.close()
        return name
    else :
        return False

def gen(recognition):
    global jarvis_mi, count, face_id
    i = 0
    while True:
        i = i+1
        frame = recognition.get_test()
        # 실시간 영상 송출
        yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        if i == 30:
            break 
    
    while True:  
        frame, label = recognition.get_frame()
        jarvis_mi = label
        # 실시간 영상 송출
        yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n' )
        # 리턴 받은 label을 이용하여 DB조회, 사용자 인증
        sr = sqlResult(jarvis_mi)
        
        # 사용자 인증시 결과 전송
        if (sr != "") and (sr != " "):
            result = json.dumps({'result': True, 'userid': jarvis_mi, 'username': sr })
            socket_client.emit('auth-data', result)
            """ recognition.__del__()
            exit() """
            
@app.route('/video_feed')
def video_feed(): 
    return Response(gen(FaceRecognition()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 서버 ip주소와 포트 
    socket_server.run(app,host,port)
