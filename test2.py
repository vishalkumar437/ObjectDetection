import cv2
import numpy as np
from tkinter import *
from tkinter import messagebox
import time
from playsound import playsound
from twilio.rest import Client
import datetime
from PIL import Image,ImageTk

global my_text
my_text='Updated..!'

asid='AC03578d4af06270afe36ab839878b58e4'
authtkn='b2ea172edf0ada93c5a4c237c3cd3a0d'

tphone='+19035687171'
myphn='+918789658374'

cap = cv2.VideoCapture(0)
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3
flag=1
flag_msg=0
client = Client(asid,authtkn)
c_time= datetime.datetime.now()

root = Tk()
root.title('Set Project - Animal Detector')
frame = Frame(root)
frame.pack()
L1=Label(frame)
L1.pack()

classesFile = 'coco.names'

classNames = []
with open(classesFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
# print(classNames)

modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
def findObjects(outputs,img):
    hT,wT,cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence>confThreshold:
                w,h = int(det[2]*wT),int(det[3]*hT)
                x,y = int((det[0]*wT) - w/2), int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
   
    for i in indices:
        print('**')
        #print(i)
        i = i.item(0)
        box = bbox[i]
        print(i)
        #print(box)
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()}{int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
        stre=classNames[classIds[i]]
        if(stre == 'cow'):
            detected(stre)
            
def flg():
    if(flag==1):
        listofgl=globals()
        listofgl['flag']=0
        w2['text']='Alarm off'
        w2.config()
        w2.update()
        w2.pack()
    else:
        listofgl=globals()
        listofgl['flag']=1
        w2['text']='Alarm on'
        w2.config()
        w2.update()
        w2.pack()
    
def detected(stre):
    w3['text']='Cow has been detected'
    w3.config()
    w3.update()
    w3.pack()
    if(flag==1):
        message = client.messages.create(
        body="Cow has been detected in farm",
        from_=tphone,
        to=myphn
        )
        print(message.body)
        playsound('abc.wav')
        time.sleep(4.5)
    

def loo():
    w3['text']='Detection in process..'
    w3.config()
    w3.update()
    w3.pack()
    while True:
        success, img = cap.read()
        imga = cap.read()
        imga = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        imga = ImageTk.PhotoImage(Image.fromarray(imga))
        L1['image']=imga
        
        blob = cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop = False)
        net.setInput(blob)

        layerNames = net.getLayerNames()
        #print(layerNames)
        outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)
        
        root.update()

        findObjects(outputs,img)
                
        cv2.imshow('Image',img)
        cv2.waitKey(1)
        
if(flag==0):
    w2['text']='Alarm off'
    w2.config()
    w2.update()
    w2.pack()



w3 = Canvas(root, width=1000, height=50)
w3.pack()
canvas_height=50
canvas_width=1000
y = int(canvas_height / 6)
w3.create_line(0, y, canvas_width, y )
w3 = Label(root, text='Code Not Started')
w3.pack()
w2 = Canvas(root, width=1000, height=50)
w2.pack()
w2.create_line(0, y, canvas_width, y )
w2 = Label(root, text='Alarm On')
w2.pack()
redbutton = Button(frame, text = 'Start', fg ='red', command=loo)
redbutton.pack( side =TOP)
redbtn = Button(frame, text = 'Toggle alarm', fg ='red', command=flg)
redbtn.pack( side = TOP)


mainloop()
