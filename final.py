import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)                    #video capture
face_recogniser=cv2.face.LBPHFaceRecognizer_create()
face_recogniser.read('trained_Data.yml')
id=0
name={1:"darshan",2:"Jitu",3:"Papa",4:"Priya",5:"Happy",6:"Manali",7:"Ritu"}    #dictionary of id and name
while(True):
    ret,img=cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 5)
        label, confidence = face_recogniser.predict(gray[y:y+h,x:x+w])

        print(label)
        print(confidence)
        if(confidence<90):             # confidence level
          id=name[label]                 # person dectected name from dictionry
          cv2.putText(img, id, (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    cv2.imshow("faces",img)
    if(cv2.waitKey(1)==ord("q")):
        break
cap.release()
cv2.destroyAllWindows()

