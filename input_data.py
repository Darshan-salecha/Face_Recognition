# This is to capture data and store st particular path as mentioned
# We have to give input ID in this code of unique person to train check ID in final.py file for existing ID


import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#face_classifier
cap=cv2.VideoCapture(0)
id=input("enter user id")       #ID input of person
sample=0
while cap.isOpened():
    ret,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)          #color to gray
    #cv2.imshow("darshan",img)
    faces=face_cascade.detectMultiScale(gray,1.2,5)       #face detection
    if(len(faces)==1):
        sample = sample + 1
        (x,y,w,h) =faces[0]
        # store at this path
        cv2.imwrite('C:\\Users\DELL\\Desktop\\photos\\test\\user'+str(id)+"."+str(sample)+'.jpg', gray[y:y+h,x:x+w])
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.waitKey(1)
        if sample>30:     #number of img store
          break
    cv2.imshow('faces', img)
cap.release()
cv2.destroyAllWindows()