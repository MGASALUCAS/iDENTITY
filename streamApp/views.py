from django.shortcuts import render
from django.http.response import StreamingHttpResponse

from streamApp import camera
from streamApp.camera import gen_frames
from flask import Flask, render_template, Response

import importlib


# Create your views here.
def index(request):
    # importlib.import_module('hello')
    # exec(open('camera.py').read())
    # import runpy
    # runpy.run_path(path_name='camera.py')
    importlib.reload(camera)
    if True:
        # cv2.imshow('Video', frame)
        # ret, buffer = cv2.imencode('.jpg', frame)
        # img = buffer.tobytes()
        # yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')  # concat frame one by one and
        # # show result

        # app = Flask(__name__)
        #
        # @app.route('/video_feed')
        # def video_feed():
        #     # Video streaming route. Put this in the src attribute of an img tag
        #     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
        #
        # @app.route('/')
        # def index():
        #     # """Video streaming home page."""
        #     return render_template('index.html')
        #
        # if __name__ == '__main__':
        #     app.run(debug=True)

        return render(request, 'index.html')


def home(request):
    return render(request, 'home.html')


def gen(camera):
    while True:
        frame = camera.gen_frames()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def video_stream(request):
    return StreamingHttpResponse(gen_frames(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')
