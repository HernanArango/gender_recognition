from local_binary_patterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
#import argparse
#opencv
import cv2
 
# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []


# loop over the training images
for imagePath in paths.list_images("faces/training"):
	# load the image, convert it to grayscale, and describe it
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)
 
	# extract the label from the image path, then update the
	# label and data lists
	labels.append(imagePath.split("/")[-2])
	data.append(hist)
 
# train a Linear SVM on the data
model = LinearSVC(C=100.0, random_state=42)
model.fit(data, labels)