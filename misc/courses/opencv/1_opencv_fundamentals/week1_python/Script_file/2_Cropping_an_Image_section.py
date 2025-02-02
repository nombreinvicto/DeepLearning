# Import libraries
import cv2
import numpy as np
from dataPath import DATA_PATH

# Read image
image = cv2.imread(DATA_PATH+"images/boy.jpg")


# Crop out a rectangle
crop = image[40:200,170:320]
cv2.imwrite("results/croppedImage.png",crop)

cv2.imshow("Cropped Image",crop)

while True:
    c = cv2.waitKey(20)
    if c == 27:
        break

cv2.destroyAllWindows()
