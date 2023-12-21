import cv2
import numpy as np
import time
import HandTrackingModule as htm

color = (125, 255, 126)
bt = 15
xp, yp = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4,720)
detector = htm.handDetector(min_detection_confidence=0.85)
imgcanvas = np.zeros((720,1280,3),np.uint8)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img,label,num = detector.findHands(img)
    if(num==0):
        lmlist=detector.findPosition(img)
    elif(num>0):
        lmlistR = detector.findPosition(img, handNo=0)
        lmlistL = detector.findPosition(img, handNo=1)

    if len(lmlist) != 0 and num==0 :
        fingers = detector.fingerUp(lmlist,label)
        print(fingers,label,num)
        if label == 'Right' :
            x1, y1 = lmlist[8][1:]  # Extracting coordinates of index 8
            x2, y2 = lmlist[12][1:]  # Extracting coordinates of index 12
            if xp ==0 and yp ==0:
                xp,yp=x1,y1

            if fingers[1] and fingers[2] :
                color =(0,0,0)
                bt = 30
                cv2.circle( img,(x1,y1),bt,color,cv2.FILLED)
                cv2.line(imgcanvas,(xp,yp),(x1,y1),color,bt)
                xp,yp = x1,y1
            if fingers[1] and fingers[2] == False  :
                color =(125, 255, 126)
                bt = 15
                cv2.circle(img,(x1,y1),bt,color,cv2.FILLED)
                cv2.line(imgcanvas,(xp,yp),(x1,y1),color,bt)
                xp,yp = x1,y1
            if fingers[1] and fingers[2] and fingers[3] and fingers[4]  :
                imgcanvas = np.zeros((720,1280,3),np.uint8)
            if fingers[1]==False and fingers[2] == False and fingers[3] == False and fingers[4] == False and fingers[0] == False:
                xp,yp = 0,0    
    elif num==0:
        xp,yp = 0,0
    elif len(lmlistR) != 0 and len(lmlistL) != 0  and num>0 :
        fingers_r = detector.fingerUp(lmlistR,label="Right")
        fingers_left = detector.fingerUp(lmlistL,label="Left")
        print(fingers,fingers_left,label,num)
        if fingers_r[1] and fingers_left[1]:
            print("")
   
    imggray = cv2.cvtColor(imgcanvas,cv2.COLOR_BGR2GRAY)
    _, imginv = cv2.threshold(imggray,50,255,cv2.THRESH_BINARY_INV)
    imginv = cv2.cvtColor(imginv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imginv)
    img = cv2.bitwise_or(img,imgcanvas)
    cv2.imshow('frame', img)
    # cv2.imshow('Img',imgcanvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
