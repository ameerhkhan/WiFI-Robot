from django.shortcuts import render
from Objector.models import Detector
from Objector.detection_script import detection_func
import cv2
import base64
import numpy as np
import pandas as pd
# from django.http import HttpResponse.StreamingHttpResponse

# Create your views here.

# Maybe the stream needs to be here?
def home(request):
    return render(request, 'index.html')

def detection_view(button=False):
    all_detections = []
    total_detections = {}
    # while button == True:
    frame, detections = detection_func()
    cv2.imwrite("frame.jpg", frame)
    # cv2.imshow(frame)
    # all_detections.append(detections)


    values = pd.Series(detections).value_counts().index
    count = pd.Series(detections).value_counts().values

    for key, value in zip(values, count):
        total_detections[key] = value
    
    return (frame, total_detections)

def update_frame(frame):
    # encode into jpeg format.
    flag, enc_frame = cv2.imencode(".jpg", frame)

    # if not flag:
    #     continue

    #base64 frame encoding.
    frame_b64 = base64.b64encode(enc_frame)
    # yield(b'--frame\r\n'
    #       b'Content-TypeL image/jpg\r\n\r\n' + enc_frame + b'\r\n\r\n')
    return(frame_b64)


def objector_video(request):
    detections = Detector.objects.all()
    # start_button = int(request.GET['START'])
    start_temp = True
    
    frame, detections_main = detection_view(start_temp)
    encoded_frame = update_frame(frame)


    context = {
        'detections': detections,
        'encoded_frame': encoded_frame,
    }

    return render(request, 'objector_video.html', context)