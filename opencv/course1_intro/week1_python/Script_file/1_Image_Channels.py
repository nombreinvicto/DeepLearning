# Import libraries
import cv2
import numpy as np
from dataPath import DATA_PATH

# Path of the image to be loaded
# Here we are supplying a relative path
imagePath = DATA_PATH + "/images/musk.jpg"

# Read the image
img = cv2.imread(imagePath)

# Display image
#cv2.imshow("Image",img)

# Convert BGR to RGB colorspace
imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#cv2.imshow("Image RGB",imgRGB)
cv2.imwrite("results/imgRGB.png",imgRGB)

cv2.imwrite("results/blueChannel.png",img[:,:,0])
cv2.imwrite("results/greenChannel.png",img[:,:,1])
cv2.imwrite("results/redChannel.png",img[:,:,2])

cv2.imshow("Image BGR",img)
cv2.imshow("Image blue",img[:,:,0])
cv2.imshow("Image green",img[:,:,1])
cv2.imshow("Image red",img[:,:,2])

while True:
    c = cv2.waitKey(20)
    if c == 27:
        break

cv2.destroyAllWindows()