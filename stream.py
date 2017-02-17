# main.py

from flask import Flask, Response
from camera import VideoCamera

app = Flask(__name__)
PORT = 5800  # port that the server runs on
# available ports are 5800-5810


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed/<num>')
def video_feed(num=None):
    return Response(gen(VideoCamera(int(num))),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True, port=PORT)
