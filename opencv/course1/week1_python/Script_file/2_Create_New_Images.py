# Import libraries
import cv2
import numpy as np
from dataPath import DATA_PATH

# Read image
image = cv2.imread(DATA_PATH+"images/boy.jpg")

# Create a new image by copying the already present image using the copy operation
imageCopy = image.copy()

# Create an empty matrix
emptyMatrix = np.zeros((100,200,3),dtype='uint8')
cv2.imwrite("results/emptyMatrix.png",emptyMatrix)

emptyMatrix = 255*np.ones((100,200,3),dtype='uint8')
cv2.imwrite("results/emptyMatrix1.png",emptyMatrix)

emptyOriginal = 100*np.ones_like(image)
cv2.imwrite("results/emptyOriginal.png",emptyOriginal)


cv2.imshow("Empty matrix",emptyMatrix)
cv2.imshow("Empty Original",emptyOriginal)
cv2.imshow("Original Image",image)

while True:
    c = cv2.waitKey(20)
    if c == 27:
        break

cv2.destroyAllWindows()
