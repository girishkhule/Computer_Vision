from sklearn.preprocessing import LabelEncoder #ncode target labels with value between 0 and n_classes-1.
from sklearn.svm import SVC #Support Vector Classification.
import pickle #Create portable serialized representations of Python objects.

#initilizing of embedding & recognizer
embeddingFile = "output/embeddings.pickle"
#New and Empty at initial
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"

print("Loading face embeddings...")
data = pickle.loads(open(embeddingFile, "rb").read())

print("Encoding labels...")
labelEnc = LabelEncoder()
labels = labelEnc.fit_transform(data["names"]) #Fit label encoder and return encoded labels.


print("Training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels) #Fit the SVM model according to the given training data.

f = open(recognizerFile, "wb")
f.write(pickle.dumps(recognizer))
f.close()

f = open(labelEncFile, "wb")
f.write(pickle.dumps(labelEnc))
f.close()
print("Completed...")