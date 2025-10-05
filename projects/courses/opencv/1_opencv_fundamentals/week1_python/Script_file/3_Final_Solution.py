# Import libraries
import cv2
import numpy as np
from dataPath import DATA_PATH

# Read image
image = cv2.imread(DATA_PATH+"images/boy.jpg")

brightnessOffset = 50

# Add the offset for increasing brightness
brightHighOpenCV = cv2.add(image, np.ones(image.shape,dtype='uint8')*brightnessOffset)

brightHighInt32 = np.int32(image) + brightnessOffset
brightHighInt32Clipped = np.clip(brightHighInt32,0,255)

# Display the outputs
cv2.imwrite("results/brightHighOpenCV.png",brightHighOpenCV)
cv2.imwrite("results/brightHighInt32Clipped.png",brightHighInt32Clipped)

# Add the offset for increasing brightness
brightHighFloat32 = np.float32(image) + brightnessOffset
brightHighFloat32NormalizedClipped = np.clip(brightHighFloat32/255,0,1)

brightHighFloat32ClippedUint8 = np.uint8(brightHighFloat32NormalizedClipped*255)

# Display the outputs
cv2.imwrite("results/brightHighFloat32NormalizedClipped.png",brightHighFloat32NormalizedClipped)
cv2.imwrite("results/brightHighFloat32ClippedUint8.png",brightHighFloat32ClippedUint8)

cv2.imshow("Bright High Float32 Normalized Clipped",brightHighFloat32NormalizedClipped)
cv2.imshow("Bright High Float32 Clipped",brightHighFloat32ClippedUint8)

while True:
    c = cv2.waitKey(20)
    if c == 27:
        break

cv2.destroyAllWindows()
