# Import cv2 module
import cv2
from dataPath import DATA_PATH

# Path to the image we are going to read
# This can be an absolute or relative path
# Here we are using a relative path
imageName = DATA_PATH+"images/boy.jpg"

# Load the image
image = cv2.imread(imageName, cv2.IMREAD_COLOR)

cv2.imshow("Image",image)

while True:
    c = cv2.waitKey(20)
    if c == 27:
        break

cv2.destroyAllWindows()
