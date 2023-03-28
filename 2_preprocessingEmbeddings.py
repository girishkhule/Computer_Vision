from imutils import paths
import numpy as np
import imutils #for resize
#Pickling is the serializing and de-serializing 
#of python objects to a byte stream. Unpicking is the opposite.
#Pickling is used to store python objects. 
#This means things like lists, dictionaries, class objects, and more
import pickle
import cv2
import os

dataset = "dataset"

#initial name for embedding file
embeddingFile = "output/embeddings.pickle"

#initializing model for embedding Pytorch
embeddingModel = "openface_nn4.small2.v1.t7" 

#initialization of caffe model for face detection
prototxt = "model/deploy.prototxt"
model =  "model/res10_300x300_ssd_iter_140000.caffemodel"

#loading caffe model for face detection
#detecting face from Image via Caffe deep learning
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

#loading pytorch model file for extract facial embeddings
#extracting facial embeddings via deep learning feature extraction
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

#gettiing image paths
imagePaths = list(paths.list_images(dataset))

#initialization
knownEmbeddings = []
knownNames = []
total = 0
conf = 0.4

#we start to read images one by one to apply face detection and embedding
for (i, imagePath) in enumerate(imagePaths):
	print("Processing image {}/{}".format(i + 1,len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2] #name for labelling
	image = cv2.imread(imagePath) 
	image = imutils.resize(image, width=900)
	(h, w) = image.shape[:2]  #height and width
	#converting image to blob for dnn face detection
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)

	#setting input blob image
	detector.setInput(imageBlob)
	#prediction the face
	detections = detector.forward()

	if len(detections) > 0:
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2] #face presence or not if no face its 0

		if confidence > conf: #The optional threshold to filter weak face detections.
			#ROI (range of interest)
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) 
			(startX, startY, endX, endY) = box.astype("int")
			face = image[startY:endY, startX:endX]  #face part is cropped 
			(fH, fW) = face.shape[:2]

			if fW < 20 or fH < 20: #check for single image in image,if two img. two rect will be their
				continue
			#image to blob for face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
			#facial features embedder input image face blob
			embedder.setInput(faceBlob)
			vec = embedder.forward()
			knownNames.append(name)
			# add the name of the person + corresponding face
			# embedding to their respective lists
			knownEmbeddings.append(vec.flatten())
			total += 1

print("Embedding:{0} ".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(embeddingFile, "wb") #write binary
f.write(pickle.dumps(data)) # pickle.dump() to put the data into opened file.
f.close()
print("Process Completed")