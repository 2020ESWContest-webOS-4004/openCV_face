#**
#*  service         :   main.py
#*  type            :   python
#*  date            :   2020.09.03
#*  author          :   한지훈(RORA)
#*  description     :   플라스크 서버
#**

from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit, send
# from camera import VideoCamera
from recognition import FaceRecognition
import json
import os

#newCode="sudo modprobe bcm2835-v4l2"
#os.system(newCode)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
host = '0.0.0.0'
port = '5000'
found = []

@app.route('/')
def index():
    # rendering webpage
    return render_template('index.html')


def messageReceived(methods=['GET', 'POST']):
    print('message was received!!!')

@socketio.on('my event')
def handle_my_custom_event(json, methods=['GET', 'POST']):
    print('received my event: ' + str(json))
    socketio.emit('my response', json, callback=messageReceived)

def face_result(result):
    if result == 555:
        return print('Recognition')
    elif result == 333:
        return print('Not Recognition')

def gen(recognition):
    while True:
        #get camera frame
        frame, f = recognition.get_frame()
        found = recognition.face_found(f)
        #send_data(face_result(found))
        yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    

@app.route('/video_feed')
def video_feed():
    return Response(gen(FaceRecognition()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def video_result():
    return Response(gen_face(FaceRecognition()))


if __name__ == '__main__':
    # defining server ip address and port
    socketio.run(app,host,port)
