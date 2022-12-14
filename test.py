# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 18:11:59 2022

@author: b4100
"""

import numpy as np
import argparse
import cv2

'''
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
'''

prototxt="deploy.prototxt"
model="res10_300x300_ssd_iter_140000.caffemodel"
Confidence=0.5

vid = cv2.VideoCapture(0)

while 1:
    ret, frame = vid.read()
    
    image=frame
    
    
    
    net = cv2.dnn.readNetFromCaffe(prototxt,model)  #prototxt model
    #image = cv2.imread(image)  # image
    
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
    
    
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > Confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
            (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow("Output", image)
    #cv2.waitKey(0)
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
