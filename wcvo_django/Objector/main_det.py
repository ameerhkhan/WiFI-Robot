# This file is not required.


from detection_script import detection_func
import numpy as np
from Objector.models import Detector


# A button to initiate the stream
button = False

all_detections = []

while button == True:
    frame, detections = detection_func()
    cv2.imshow(frame)
    all_detections.append(detections)


values = pd.Series(detections).value_counts().index
count = pd.Series(detections).value_counts().values

for key, value in zip(values, count):
    total_detections[key] = value


df = pd.DataFrame.from_dict(total_detections, orient='index', columns=['COUNT'])
df.to_csv("live_detections.csv")







