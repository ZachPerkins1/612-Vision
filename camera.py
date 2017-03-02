# camera.py

import cv2


class VideoCamera(object):
    def __init__(self, id):
        self.video = cv2.VideoCapture(id)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()

        # opencv doesn't default to jpg
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
