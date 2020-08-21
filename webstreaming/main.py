from flask import Flask, render_template, Response
from flask_socketio import SocketIO
# from camera import VideoCamera
from recognition import FaceRecognition

app = Flask(__name__)
app.config['SECRET_KEY'] = 'BCODE_Flask'
socketio = SocketIO(app)

@app.route('/')
def index():
    # rendering webpage
    return render_template('index.html')

def gen(recognition):
    while True:
        #get camera frame
        frame = recognition.get_frame()
        yield (b'--frame\r\n'
                + b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(FaceRecognition()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # defining server ip address and port
    socketio.run(app,host='0.0.0.0',port='5000')