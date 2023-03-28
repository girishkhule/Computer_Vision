from collections import Iterable
import numpy as np
import imutils
import pickle
import time
import cv2
import csv
from datetime import datetime

filename = "attendence.csv"
# opening the file with w+ mode truncates the file
f = open(filename, "w+")
f.writelines(f'\nName,Roll_Number,Date_Stamp')
f.close()

# now = datetime.now()
# datestring = now.strftime('%D:%H:%M')

# f.writelines(f'\nName,Roll_Number,Date_Stamp')

def takeAttendence(text):
    with open('attendence.csv', 'r+') as f:
        mypeople_list = f.readlines()
        nameList = []
        for line in mypeople_list:
            entry = line.split(',')
            nameList.append(entry[0])
           
        if name not in nameList:
            now = datetime.now()
            # datestring = now.strftime('%D/%M:%H:%M')
            datestring = now.strftime('%D:%H:%M')
            f.writelines(f'\n{name},{Roll_Number},{datestring}')

def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


embeddingFile = "output/embeddings.pickle"
embeddingModel = "openface_nn4.small2.v1.t7"
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"
# conf = 0.5
conf = 0.6

print("[INFO] loading face detector...")
prototxt = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

recognizer = pickle.loads(open(recognizerFile, "rb").read()) #load read binary
le = pickle.loads(open(labelEncFile, "rb").read()) #le = label reader

Roll_Number = ""
box = []
print("[INFO] starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(2.0)

t_end = time.time() + 15 * 1 # This will run for 1 min x 60 s = 60 seconds.

while time.time() < t_end:
#while True:
    _, frame = cam.read()
    frame = imutils.resize(frame, width=500)
    (h, w) = frame.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(imageBlob)  #face detection
    detections = detector.forward()

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > conf:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20: #multiple identification
                continue
            # construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            with open('student.csv', 'r') as csvFile:
                reader = csv.reader(csvFile)
                for row in reader:
                    box = np.append(box, row)
                    name = str(name)
                    if name in row:
                        person = str(row)
                        print(name)
                listString = str(box)
                print(box)
                if name in listString:
                    singleList = list(flatten(box))
                    listlen = len(singleList)
                    Index = singleList.index(name)
                    name = singleList[Index]
                    Roll_Number = singleList[Index + 1]
                    print(Roll_Number)
                else:
                    name = "Unknown"
                    Roll_Number = "Unknown"
                    print(Roll_Number)

            text = "{} : {} : {:.2f}%".format(name, Roll_Number, proba * 100)
            if proba * 100 > 97 :
                takeAttendence(text)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)



    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    # key = cv2.waitKey(5000) 
 

    if key == 27:
        break


cam.release()
cv2.destroyAllWindows()