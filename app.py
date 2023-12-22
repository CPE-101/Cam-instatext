from flask import Flask, render_template, Response, jsonify, redirect
from flask_socketio import SocketIO
import cv2
from fastai.vision.all import *
from mltu.configs import BaseModelConfigs
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
from PIL import Image
import pathlib
import HandTrackingModule as htm

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def drawing_mode():
    return render_template('handtracking.html')

@app.route('/asl')
def asl_mode():
    prediction_history.clear()
    return render_template('asl_alphabet.html')

@app.route('/teaching')
def teach_mode():
    return render_template('teaching.html')

#--====================================================== Camera =======================================================--#

@app.route('/camera_drawing')
def camera_drawing():
    return Response(generate_hand_tracking_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera_asl')
def camera_asl():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

#Basic Camera Setting
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4,720)

#--====================================================== Typing Word from drawing with the hands =======================================================--#
#Video live with Hand Tracking for Drawing Mode
def generate_hand_tracking_frames():
    color = (125, 255, 126)
    bt = 15
    iswrite,candelete = False,True
    xp, yp = 0, 0
    detector = htm.handDetector(min_detection_confidence=0.85)
    imgcanvas = np.zeros((720,1280,3),np.uint8)
    last_pop_time = time.time()  
    delay_time = 1
    text=[] #For Containing the String from Prediction Text
    while True:
        #show camera live on the website
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)

        img,label,num = detector.findHands(img)
        if(num==0):
            lmlist=detector.findPosition(img)
        '''    
        elif(num>0):
            lmlistR = detector.findPosition(img, handNo=0)
            lmlistL = detector.findPosition(img, handNo=1)
        '''

        if len(lmlist) != 0 and num==0 :
            fingers = detector.fingerUp(lmlist,label)
            # print(lmlist,end="\n\n")
            # print(fingers,label,num)
            if label == 'Right' :
                x1, y1 = lmlist[8][1:]  # Extracting coordinates of index 8
                #x2, y2 = lmlist[12][1:]  # Extracting coordinates of index 12
                if xp ==0 and yp ==0:
                    xp,yp=x1,y1
                '''
                # if fingers[1] and fingers[2] :
                #     color =(0,0,0)
                #     bt = 50
                #     cv2.circle( img,(x1,y1),bt,color,cv2.FILLED)
                #     cv2.circle( imgcanvas,(x1,y1),bt,color,cv2.FILLED)
                #     cv2.line(imgcanvas,(xp,yp),(x1,y1),color,bt)
                #     xp,yp = x1,y1
                '''
                #Reseting Text
                if fingers[1] and fingers[4] and fingers[2] ==False and fingers[3] == False and fingers[0] ==False :
                    text =[]
                    update_text(text) 
                    print(''.join(map(str,text)))
                #Drawing text   
                elif fingers[1] and fingers[2] == False  :
                    iswrite,candelete = True,True
                    color =(125, 255, 126)
                    bt = 15
                    cv2.circle(img,(x1,y1),bt,color,cv2.FILLED)
                    cv2.line(imgcanvas,(xp,yp),(x1,y1),color,bt)
                    xp,yp = x1,y1
                #Deleting string
                elif fingers[1] and fingers[2] and fingers[3] and fingers[4] and candelete :
                    current_time = time.time()
                    if current_time - last_pop_time >= delay_time:
                        candelete = False
                        try: 
                            text.pop(-1)
                            update_text(text)
                            last_pop_time = current_time  #Update the last_pop_time
                            # time.sleep(1) #delay time for the loop to not update too fast (that will take a lot of ram)
                        except:
                            pass
                #Predicting text
                elif fingers[1]==False and fingers[2] == False and fingers[3] == False and fingers[4] == False and fingers[0] == False and iswrite:
                    xp,yp = 0,0    
                    iswrite,candelete = False,True
                    imgwhite = cv2.bitwise_not(imggray)
                    cv2.imwrite("pic.png",imgwhite)
                    text = CROPandpredict(text)
                    imgcanvas = np.zeros((720,1280,3),np.uint8) 
                    update_text(text) 
                #Reseting Location of Fingers
                else :
                    candelete = True
                    xp,yp = 0,0
        elif num==0 and iswrite:
            iswrite,candelete = False,True
            xp,yp = 0,0
            imgwhite = cv2.bitwise_not(imggray)
            cv2.imwrite("pic.png",imgwhite)
            text = CROPandpredict(text)
            imgcanvas = np.zeros((720,1280,3),np.uint8)
            update_text(text) 
        elif num==0:
            candelete = True
        '''
            xp,yp = 0,0     
        elif len(lmlistR) != 0 and len(lmlistL) != 0  and num>0 :
            x1 , y1 = lmlistR[8][1:]
            fingers_r = detector.fingerUp(lmlistR,label="Right")
            fingers_left = detector.fingerUp(lmlistL,label="Left")
            # print(fingers_r,fingers_left,label,num)
            #     h,w,c = img.shape
            #     curx,cury=int(x1/w*screen_width),int(y1/h*screen_width) 
            #     pyautogui.moveTo(curx, cury)
            # elif fingers_r[2] and fingers_r[1] and fingers_left[1]:  
            #     pyautogui.click()
        print(text,''.join(map(str,text)))
        '''

        #For Showing the frame with the Drawing on the Camera
        imggray = cv2.cvtColor(imgcanvas,cv2.COLOR_BGR2GRAY)
        _, imginv = cv2.threshold(imggray,50,255,cv2.THRESH_BINARY_INV)
        imginv = cv2.cvtColor(imginv,cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img,imginv)
        img = cv2.bitwise_or(img,imgcanvas)

        #For Camera on the Website
        _, buffer = cv2.imencode('.jpg', img)
        frame_bytes = buffer.tobytes()
        
        
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
         
#For Handwriting Recognition
configs = BaseModelConfigs.load("models/handwriting_recognition_model/202312162202/configs.yaml")
model = htm.ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)
df = pd.read_csv("models/handwriting_recognition_model/202312162202/val.csv").values.tolist()
image_path = 'pic.png' 

#Crop the Drawing and predict
def CROPandpredict(text) :
    temp = htm.cropping_image(image_path)
    if temp == 0:
        pass
    else:
        image = cv2.imread(image_path)
        prediction_text = model.predict(image)
        for i in prediction_text :
            text.append(i)
    return text

#Update text
@socketio.on('update_text')
def update_text(text):
    result = ''.join(map(str,text))
    socketio.emit('text_updated', {'result': result})   

#--====================================================== Typing Word from ASL-Alphabet =======================================================--#
#Normal Video live for ASL-Alphabet Mode
def generate_frames():
    while True:
        #Show camera live on the website
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)

        #For Camera on the Website
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

#Load ASL-Alphabet Recognition AI model
temppp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
asl_recognition = load_learner("models/asl_recognition_model.pkl")
pathlib.PosixPath = temppp

def predict_asl_without_bg():
    hand_region = crophand()
    if hand_region is not None:
        img = Image.fromarray(hand_region) #Convert the frame to an Image for AI prediction
        prediction, _, _ = asl_recognition.predict(img)  #Predict with AI
        return prediction
    else:
        return 100

#--====================================================== Image Segmentation For Prediction =======================================================--#

#Extract hand region from the frame
def extract_hand_region_with_margin(frame, landmarks, margin=100): #margin for add more space for crop hand pic
    xs = [int(l.x * frame.shape[1]) for l in landmarks.landmark]
    ys = [int(l.y * frame.shape[0]) for l in landmarks.landmark]

    min_x, max_x = min(xs) - margin, max(xs) + margin
    min_y, max_y = min(ys) - margin, max(ys) + margin

    #Ensure the coordinates are within the frame boundaries
    min_x = max(0, min_x)
    max_x = min(frame.shape[1], max_x)
    min_y = max(0, min_y)
    max_y = min(frame.shape[0], max_y)

    return frame[min_y:max_y, min_x:max_x]

#Crop Only Hand from the frame
def crophand():
    #Capture the current frame from the video feed 
    success, frame = cap.read()

    frame = cv2.flip(frame, 1)
    #Initialize MediaPipe Hands (For Image Segmentation)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame) #Process the image to get hand landmarks

    hand_region = None
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            hand_region = extract_hand_region_with_margin(frame, landmarks) #Extract hand region
            cv2.imshow("Hand Region", hand_region) #Display the hand region

            #Save in folder to see what's image from crophand() be like ***Used ONLY during development.***
            temppp = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
            cv2.imwrite("crophand.png", hand_region) 
            pathlib.PosixPath = temppp
    else:
        hand_region = None
        
    return hand_region

#--====================================================== Prediction History (ASL) =======================================================--#
#For contain all prediction as the text
prediction_history = [] 

#Show the word prediction
@app.route('/predict_asl', methods=['POST'])
def predict_asl():
    prediction = predict_asl_without_bg()
    if prediction == 100: #No hand
        prediction == '' 
    elif prediction == 'nothing': #Do nothing
        prediction == '' 
    elif prediction == 'space':
        prediction_history.append(' ')
    elif prediction == 'del':
        prediction_history.pop() #Delete the last prediction from the word prediction
    else:
        prediction_history.append(prediction) #Append from the last prediction
    
    return jsonify({'history': prediction_history})

#Clear the word prediction
@app.route('/clear_result', methods=['POST'])
def clear_result():
    prediction_history.clear()
    return jsonify({'history': prediction_history})

#Delete the last prediction from the word prediction
@app.route('/delete_lastprediction', methods=['POST'])
def delete_lastprediction():   
    prediction_history.pop()
    return jsonify({'history': prediction_history})

if __name__ == '__main__':
    socketio.run(app, debug=True)