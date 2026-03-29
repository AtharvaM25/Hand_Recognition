import os
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap_ob = cv2.VideoCapture(0)
dectector1 = HandDetector(maxHands=1)
offset = 25
imgSize = 300
counter = 0

while (True):
    success, img = cap_ob.read()
    hands, img = dectector1.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        y1, y2 = max(0, y), min(y+h, img.shape[0])
        x1, x2 = max(0, x), min(x+w, img.shape[1])
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y1-offset:y2+offset, x1-offset:x2+offset]
        aspectRatio = h/w
        if (aspectRatio > 1):
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:,
                     wGap:wCal+wGap] = imgResize
        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap,
                     :] = imgResize

        if imgCrop.size > 0:
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(5)

    if key == ord("s"):
        counter += 1
        timestamp = int(time.time())
        cv2.imwrite(
            f"C:/Users/athar/Projects/Images for HandGesture/Images_V2/Y/image_{counter}_{timestamp}.png", imgWhite)
        print(counter)
