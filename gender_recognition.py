from local_binary_patterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import pickle
#opencv
import cv2
 
# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(24, 8)


model = pickle.load(open("model.save", 'rb'))


# loop over the testing images
for imagePath in paths.list_images("faces/testing"):
	# load the image, convert it to grayscale, describe it,
	# and classify it
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)
	prediction = model.predict([hist])[0]
 
	# display the image and the prediction
	cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (0, 0, 255), 3)
	cv2.imshow("Image", image)
	cv2.waitKey(0)
