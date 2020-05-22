import  os
import numpy as np
import cv2


face_recongnizer = cv2.face.LBPHFaceRecognizer_create()
path='C:\\Users\\DELL\\Desktop\\photos\\test'              #Folder path of imgs stored to access
def getImageWith(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    print(imagePaths)
    faces = []
    faceID = []
    for imagePath in imagePaths:
       faceImg=cv2.imread(imagePath,0)
       facenp=np.array(faceImg,'uint8')
       ID=int(os.path.split(imagePath)[-1][4])     #this give ID after 'user' string
       #print(ID)
       faces.append(facenp)
       faceID.append(ID)
       #cv2.imshow("training",facenp)
       #cv2.waitKey(10)
    return faces,faceID

faces,faceID=getImageWith(path)
print(faceID)
face_recongnizer.train(faces,np.array(faceID))
face_recongnizer.save('trained_data.yml')         #trained file
cv2.destroyAllWindows()
