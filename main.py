import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*
model=YOLO('yolov8s.pt')



def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('peoplecount1.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0

area1=[(252,370),(206,372),(480,476),(513,466)]
area2=[(201,378),(180,383),(433,482),(465,479)]
while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
#    frame=cv2.flip(frame,1)
    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.boxes
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]
             
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'person' in c: 
           cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
           cv2.putText(frame,str(c),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,(0.5),(255,255,255),1)
                
        
      
            
            
        
        
            

    cv2.polylines(frame,[np.array(area1,np.int32)],True,(255,0,0),2)
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(255,0,0),2)
    cv2.putText(frame,('1'),(338,409),cv2.FONT_HERSHEY_COMPLEX,(0.6),(0,0,255),1)
    cv2.putText(frame,('2'),(284,428),cv2.FONT_HERSHEY_COMPLEX,(0.6),(0,0,255),1)


    cv2.imshow("RGB", frame)
    if cv2.waitKey(0)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()

