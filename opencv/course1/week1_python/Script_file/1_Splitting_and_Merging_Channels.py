# Import libraries
import cv2
import numpy as np
from dataPath import DATA_PATH

# Path of the image to be loaded
# Here we are supplying a relative path
imagePath = DATA_PATH + "/images/musk.jpg"

# Read the image
img = cv2.imread(imagePath)

# Split the image into the B,G,R components
b,g,r = cv2.split(img)

# Show the channels
cv2.imwrite("results/blueChannel.png",b)
cv2.imwrite("results/greenChannel.png",g)
cv2.imwrite("results/redChannel.png",r)

# Merge the individual channels into a BGR image
mergedOutput = cv2.merge((b,g,r))

# Show the merged output
cv2.imwrite("results/mergedOutput.png",mergedOutput)

cv2.imshow("Image BGR",img)
cv2.imshow("Image blue",b)
cv2.imshow("Image green",g)
cv2.imshow("Image red",r)

while True:
    c = cv2.waitKey(20)
    if c == 27:
        break

cv2.destroyAllWindows()
