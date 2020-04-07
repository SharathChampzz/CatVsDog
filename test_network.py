##python test_network.py --model M:\pythonD\DogCat\CatDog.model --image M:\pythonD\DogCat\dog.jpg
##python test_network.py --model M:\pythonD\DogCat\CatDog.model --image M:\pythonD\DogCat\cat.jpg
# import the necessary packages
import warnings
warnings.simplefilter("ignore")
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
from os import listdir

##Required Changes && while entering in cmd Prompt Give proper location of model and image.
ht=28
wd=28
pathh = "M:/pythonD/DogCat/CatDog"  # Database To take automatically the req class names.


className = []
classNames = listdir(pathh)
totClass = len(classNames)
print(classNames)
print(totClass)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
orig = image.copy()

# pre-process the image for classification
try:
    image = cv2.resize(image, (ht, wd))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
except Exception as e:
    print("Error Occured : ",e)
    


# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])

# classify the input image
# Creating Probability Tuples
##prob = []
##for i in range(0,totClass):
##    values = str(i) + 'x'
##    prob.append(values)
##prob = [zero, one,two,three,four,five,six]
##    probTuple  = (zero, one,two,three,four,five,six)
##probTuple = tuple(prob)
(zero, one) = model.predict(image)[0]
prob = [zero,one]
maxProb = max(prob)
maxIndex = prob.index(maxProb)
label = classNames[maxIndex]
proba = maxProb

for i in range(0,totClass):
    print(f'{classNames[i]} : {prob[i]}')
# build the label
label = "{}: {:.2f}%".format(label, proba * 100)

# draw the label on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
