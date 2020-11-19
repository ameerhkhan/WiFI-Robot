import numpy as np
import imutils
import time
import cv2
import os
import pandas as pd
from imutils.video import VideoStream
from Objector.models import Detector


def detection_func():
    # let's first load our model.
    print("Loading the model.")
    yolo_path = "Q:/Hamza/Python/OpenCV/object_detection/realtime_OD/YOLOv3/yolo-coco"
    confidence = 0.6
    threshold = 0.3

    # let's now load our class labels and set random color for each
    labels_path = os.path.sep.join([yolo_path, "coco.names"])
    LABELS = open(labels_path).read().strip().split("\n")

    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    # derive the path for YOLO weights and model configuration
    weights_path = os.path.sep.join([yolo_path, "yolov3.weights"])
    config_path = os.path.sep.join([yolo_path, "yolov3.cfg"])

    # load our YOLO Object detector trained on COCO Dataset.
    # Need atleast OpenCV version 3.4.2
    print("loading YOLO from disk")
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    # determine only the output layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1   ] for i in net.getUnconnectedOutLayers()]

    # Now let's start the video stream
    print("Starting the video stream")
    # vs = VideoStream(src=0).start()
    vs = cv2.VideoCapture(0)             # if using RPI use, vs = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)

    (H, W) = (None, None)

    total_detections = {}
    detections = []
    bounding_boxes = []

    while True:
        # grab a frame and resize it
        grabbed, frame = vs.read()

        if not grabbed:
            print("Not Grabbed")
            break
        frame = imutils.resize(frame, width=400)
        
        # grab the frame dimensions and convert to a blob
        (H, W) = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)

        # pass the blob through a network and get the predictions.
        net.setInput(blob)
        layer_outputs = net.forward(ln)

        # initialize our list of bboxes and confidences as well as prediction class labels.
        boxes = []
        confidences = []
        
        class_ids = []
        detections_live = []

        for output in layer_outputs:
            for detection in output:

                # extract the class id and confidence
                scores = detection[5:]
                class_id = np.argmax(scores)
                detector_confidence = scores[class_id]

                # filter out the weak predictions
                if detector_confidence > confidence:
                    # scale the bbox co-ordinates keeping in mind
                    # that YOLO returns the center co-ord of object
                    # followed by their width and height.
                    box = detection[0:4] * np.array([W, H, W, H])
                    (center_x, center_y, width, height) = box.astype("int")

                    # calculate the top and left corner of the bbox
                    x = int(center_x - (width/2))
                    y = int(center_y - (height/2))

                    # update our list of bbox co-ords, confidences and the prediction LABEL
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(detector_confidence))
                    class_ids.append(class_id)
                    detections.append(LABELS[class_id])
                    Detector.accuracy_score = detector_confidence
                    Detector.detected_object = LABELS[class_id]
                    Detector.bounding_box = (x,y,int(width), int(height))
                    print(LABELS[class_id])

        # now we will apply non-maxima suppression
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

        # ensure atleast one detection exists
        if len(idxs) > 0:
            # loop over the indexes
            for i in idxs.flatten():
                #extract the bounding box co-ordinates.
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw the box rectangle around the object
                color = [int(c) for c in COLORS[class_ids[i]]]
                cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
                text = "{}: {:.4f}".format(LABELS[class_ids[i]], confidences[i])
                cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # cv2.imshow("DETECT", frame)
                bounding_boxes.append((x,y,x+w,y+h))
        # cv2.imshow('DETECT', frame) 
        return(frame, detections)