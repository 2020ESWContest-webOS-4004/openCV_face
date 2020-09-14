#**
#*  service         :   main.py
#*  type            :   python3
#*  date            :   2020.09.14
#*  author          :   한지훈(RORA)
#*  description     :   플라스크 서버
#**

from flask import Flask, render_template, Response,request
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
count = 0 
face_id = 0 
face_name = ''

@app.route('/')
def index():
    # rendering webpage
    return render_template('index.html')

def messageReceived(methods=['GET', 'POST']):
    print('message was received!!!')

def get_json(l):
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
    return face_id

def gen(recognition):
    global face_name, count, face_id
    while True:
        frame, label = recognition.get_frame()
        count = count + 1
        if count == 10:
            face_name = label
            break
        face_id = get_json(face_name)
        
        yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n' )

@app.route('/video_feed')
def video_feed():
    return Response(gen(FaceRecognition()), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('my event')
def handle_my_custom_event(test):
    global face_name, face_id
    print('received my event: ' + str(test))
    #print(json.dumps({face_id : face_name}))
    socketio.emit('my response', json.dumps({face_id : face_name}) )
   
if __name__ == '__main__':
    # defining server ip address and port
    socketio.run(app,host,port)
