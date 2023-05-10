from django.shortcuts import render
from django.http.response import StreamingHttpResponse

from streamApp import camera
from streamApp.camera import gen_frames
from flask import Flask, render_template, Response

import importlib


# Create your views here.
def index(request):  
      importlib.reload(camera)
      while True:
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

def records(request):
    return render(request, 'charts/index.html')

def tables(request):
    return render(request, 'charts/tables.html')