from cv2 import cv2
from imutils import paths
import os

sourcePath = r"C:\Users\mhasa\Desktop\temp_cad_threshinv"
targetPath = r"C:\Users\mhasa\Desktop\mvcnn_thresh_clean\Cylindrical"

imagePaths = list(paths.list_images(sourcePath))

# button enums
ESC = 27
ENTER = 13

for image_no, path in enumerate(imagePaths):
    # retrieve the image name
    imageName = path.split(os.sep)[-1]

    # read the image
    img = cv2.imread(path)

    # show the image
    cv2.imshow('Image', img)
    clicked_button = cv2.waitKey(0) & 0xFF

    # check the button value
    if clicked_button == ESC:
        print(f"Escaping image: {image_no + 1}")

    elif clicked_button == ENTER:
        # start deletion of the file
        try:
            os.remove(os.path.join(targetPath, imageName))
            print(f"Deleting image: {image_no + 1}")
        except Exception as msg:
            print(f"Deletion failed. {msg}")
