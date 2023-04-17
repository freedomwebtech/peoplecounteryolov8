import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO


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
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            


    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()

