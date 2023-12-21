import cv2
import typing
import numpy as np
import mediapipe as mp
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer
class handDetector():
    def __init__(self, mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mode = mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.tipIds = [4,8,12,16,20]
        self.label = None
        self.num = 0
    def findHands(self, img, draw = True) :
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        # print(result.multi_hand_landmarks)
        if self.result.multi_hand_landmarks :
            handedness = self.result.multi_handedness
            for self.num,(handLand,hand_info) in enumerate( zip(self.result.multi_hand_landmarks,handedness)) :
                self.label = hand_info.classification[0].label
                # print(f"{self.label} :{handedness}")
                drawcolor = (255,0,0) if (self.label == 'Right') else (0,255,0)
                if draw :
                    self.mp_draw.draw_landmarks(img, handLand, self.mp_hands.HAND_CONNECTIONS,
                                                self.mp_draw.DrawingSpec(drawcolor,thickness=2,circle_radius=4),
                                                self.mp_draw.DrawingSpec(color=(255,255,255),thickness=2,circle_radius=4))

        return img,self.label,self.num
    def fingerUp(self,lmList,label):
        fingers = []
        if label == 'Right' :
                    # print(hand.classification[0].label,fingers_L,fingers_R)
            if len(lmList) >= max(self.tipIds[0], self.tipIds[0] - 1):
                if lmList[self.tipIds[0]][1] < lmList[self.tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                fingers.extend(self.checkfin(lmList))
        if label == "Left" :
                # print(hand.classification[0].label,fingers_L,fingers_R)
            if len(lmList) >= max(self.tipIds[0], self.tipIds[0] - 1):
                if lmList[self.tipIds[0]][1] > lmList[self.tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                fingers.extend(self.checkfin(lmList))
        return fingers
    def findPosition(self, img, handNo = 0, draw = True) :
        self.lmList=[]
        if self.result.multi_hand_landmarks :
            myHand = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                self.lmList.append([id, cx, cy])
        return self.lmList
    def checkfin(self,lmList):
        fingers=[]
        for id in range(1, 5):
                if len(lmList) >= max(self.tipIds[id], self.tipIds[id] - 2):
                    if lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:
                        fingers.append(0)
        return fingers
    
class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shape[:2][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text

def cropping_image(image_path):
    import cv2
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    try:
        max_contour = max(contours, key=cv2.contourArea)
    except:
        return 0
    x, y, w, h = cv2.boundingRect(max_contour)
    margin = 10
    x -= margin
    y -= margin
    w += 2 * margin
    h += 2 * margin
    x = max(0, x)
    y = max(0, y)
    cropped_image = image[y:y + h, x:x + w]
    cv2.imwrite('pic.png', cropped_image)
    return 1

