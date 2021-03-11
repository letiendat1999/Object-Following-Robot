# Import cac thu vien can thiet
import cv2
import numpy as np
import RPi.GPIO as GPIO     # Importing RPi library to use the GPIO pins
from time import sleep      # Importing sleep from time library to add delay in code
import math

# Dinh nghia cac gia tri
servo_pin = 18      # Initializing the GPIO 18 for servo motor
GPIO.setmode(GPIO.BCM)          # We are using the BCM pin numbering
GPIO.setwarnings(False)
GPIO.setup(servo_pin, GPIO.OUT)     # Declaring GPIO 21 as output pin
p = GPIO.PWM(servo_pin, 50)     # Created PWM channel at 50Hz frequency

confidentThresh = 0.1
nmsThresh = 0.3

# Phan main
#Phan code test
#img = cv2.imread('chuot_test.jpg')
#scale_percent = 40 # percent of original size
#width = int(img.shape[1] * scale_percent / 100)
#height = int(img.shape[0] * scale_percent / 100)
#dim = (width, height)
# resize image
#img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
p.start(7.5)
cap = cv2.VideoCapture(0)
classesFile = 'yolo.names'
classesName = []
with open(classesFile, 'rt') as f:
    classesName = f.read().rstrip('\n').split('\n')
print(classesName)
modelConfig = 'yolov3-tiny_2000.cfg'
modelWeight = 'yolov3-tiny_6000.weights'
net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeight)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObject(outputs, img):
    hT, wT, cT = img.shape
    bboxs = []
    classIDs = []
    confidents = []
    for output in outputs:
        for det in output:
            score = det[5:]
            classID = np.argmax(score)
            confident = score[classID]
            if confident > confidentThresh:
                w, h = int(det[2]*wT), int(det[3]*hT)
                x, y = int(det[0]*wT-w/2), int(det[1]*hT-h/2)
                bboxs.append([x, y, w, h])
                classIDs.append(classID)
                confidents.append(float(confident))
    indices = cv2.dnn.NMSBoxes(bboxs, confidents, confidentThresh, nmsThresh)
    for i in indices:
        i = i[0]
        box = bboxs[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x+w, w+h), (0, 255, 0), 2)
        cv2.circle(img, (int(x+w/2), int(y+h/2)), 3, (0, 255, 0), 2)
        cv2.putText(img, f'{classesName[classIDs[i]].upper()} {int(confidents[i]*100)}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 0, 255), 1)

        if int(x+w/2) >= int(img.shape[1] / 2):
            rightVal = int(x + int(w/2)) - int(img.shape[1] / 2)
            print(rightVal)
            rightFlag = 1
        
            Angle = int(round(math.acos(rightVal / math.sqrt(rightVal * rightVal + int(img.shape[1]/2) * int(img.shape[1]/2)))) * (180 / 3.14))

            cv2.putText(img, f'{rightVal}', (30, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (125, 255, 125), 1)
            print("Quay phai")
            print("Goc quay la: ", Angle)
            DutyCycle = ((Angle) / 180) + 2.5  # Tính chu kì xung
            p.ChangeDutyCycle(DutyCycle)  # Thay đổi chu kì xung
            rightFlag = 0
        else:
            leftVal = int(img.shape[1] / 2) - int(x + int(w / 2))
            print(leftVal)
            leftFlag = 1
            
            Angle = int(round(math.acos(leftVal / math.sqrt(leftVal * leftVal + int(img.shape[1]/2) * int(img.shape[1]/2)))) * (180 / 3.14))

            cv2.putText(img, f'{leftVal}', (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (125, 255, 125), 1)
            print("Quay trai")
            print("Goc quay la: ", Angle + 90)
            DutyCycle = ((Angle + 90) / 18) + 2.5  # Tính chu kì xung
            p.ChangeDutyCycle(DutyCycle)  # Thay đổi chu kì xung
            leftFlag = 0
        if int(y+h/2) >= int(img.shape[0] / 2):
            downVal = (y + int(h/2)) - int(img.shape[0] / 2)
            print(downVal)
            cv2.putText(img, f'{downVal}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (125, 255, 125), 1)

        else:
            upVal = int(img.shape[0] / 2) - (y + int(h / 2))
            print(upVal)
            cv2.putText(img, f'{upVal}', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (125, 255, 125), 1)
# ham ve cac duong thang vertical, horizontal và central point
def targetDrawing(IMG):
    hoziLine = cv2.line(IMG, (0, int(IMG.shape[0]/2)), (IMG.shape[1], int(IMG.shape[0]/2)), (255, 0, 0), 1)
    VertLine = cv2.line(IMG, (int(IMG.shape[1]/2), 0), (int(IMG.shape[1]/2), IMG.shape[1]), (255, 0, 0), 1)
    rectBox = cv2.rectangle(IMG, (int(IMG.shape[1]/2)-50, int(IMG.shape[0]/2)-50), (int(IMG.shape[1]/2)+50, int(IMG.shape[0]/2)+50), (255, 0, 0), 1)
    cenPoint = cv2.circle(IMG, (int(IMG.shape[1]/2), int(IMG.shape[0]/2)), 3, (0, 255, 0), 2)

while (cap.isOpened()):
    success, img = cap.read()
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    #Print LayerName
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    #print(outputNames)
    outPut = net.forward(outputNames)
    #print(outPut[0].shape)
    #print(outPut[1].shape)
    #print(len(outPut))
    #print(outPut[0][0])

#Call all the functions

    findObject(outPut, img)
    targetDrawing(img)

#######################################################################

    cv2.putText(img, "Right: ", (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (125, 255, 125), 1)
    cv2.putText(img, "Left: ", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (125, 255, 125), 1)
    cv2.putText(img, "Down:", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (125, 255, 125), 1)
    cv2.putText(img, "Up: ", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (125, 255, 125), 1)
    cv2.imshow('Image', img)
    if cv2.waitKey(25) == ord('q'):
        break
cap.release()
GPIO.cleanup()
cv2.destroyAllWindows()
