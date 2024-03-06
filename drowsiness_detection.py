import cv2    
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

def eye_aspect_ratio(eye):
    if len(eye) != 5:
        return 0.0
    
    # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    # Compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = np.linalg.norm(eye[0] - eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    return ear

mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')



lbl=['Close','Open']

model = load_model('models/cnnCat.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for AVI format
out = cv2.VideoWriter('output.mp4', fourcc, 8.0, (640, 480))  # Adjust the resolution (640x480) and frame rate (10.0) as needed
sound_playing = False
# Create an OpenCV window with the appropriate name
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

# Set the window properties to make it fullscreen
cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = np.argmax(model.predict(r_eye),axis=-1)  
        
        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = np.argmax(model.predict(l_eye),axis=-1)
   
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break
    # Calculate eye aspect ratio for right eye
        right_ear = eye_aspect_ratio(right_eye)
            
            # Calculate eye aspect ratio for left eye
        left_ear = eye_aspect_ratio(left_eye)
            
            # If either eye's aspect ratio is greater than or equal to a threshold, generate an alert
        if right_ear >= 0.2 or left_ear >= 0.2:
            cv2.putText(frame, "Alert! Eye aspect ratio too high", (30, height-60), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(frame,"Detection",(10,30), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(rpred[0]==0 and lpred[0]==0):
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>10):
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()
            
        except:  # isplaying = False
            pass
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    cv2.imshow('frame',frame)
    out.write(frame)
    if score < 10:
        cv2.putText(frame,"Non Drowsy",(10,30), font, 1,(255,255,255),1,cv2.LINE_AA)
        if sound_playing:
            sound.stop()
            sound_playing = False
            
    if score > 10:
        cv2.putText(frame,"Drowsy",(10,30), font, 1,(255,255,255),1,cv2.LINE_AA)
        if not sound_playing:
            sound.play()
            sound_playing = True
            
    if cv2.waitKey(1) & 0xFF ==27:
        break
if sound_playing:
    sound.stop()
cap.release()
cv2.destroyAllWindows()
out.release()