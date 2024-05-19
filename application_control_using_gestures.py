import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

import time
import subprocess
import pyautogui

class Path(object):
    import ast
    sup_programs_txt = open('sup_programs.txt', 'r')
    s_p = sup_programs_txt.read()
    paths = ast.literal_eval(s_p)
    sup_programs_txt.close()
paths = Path().paths
program_name = 'spotify'

def open_program(path_name):
    subprocess.Popen(path_name)

#from tensorflow import keras
#import keras.models
import tf_keras as keras
#from keras.models import load_model

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = keras.models.load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
#print(classNames)


# Initialize the webcam
cap = cv2.VideoCapture(0)



flag=0
while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)
    
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            # print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]

    # show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)
    


    # Show the final output
    cv2.imshow("Output", frame)
     
    if className == "peace" and flag==0:
        flag=1 
        open_program(paths[program_name])

    elif className == "smile": 
        print("Good Bye, Have a great day!") 
        break

    if flag==1 and className == "thumbs up":
        pyautogui.press('nexttrack')
        time.sleep(1.5)
    elif flag==1 and className == "thumbs down":
        pyautogui.press('prevtrack')
        time.sleep(1.5)
    elif flag==1 and className == "stop":
        pyautogui.press('playpause')
        time.sleep(1.5)

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()