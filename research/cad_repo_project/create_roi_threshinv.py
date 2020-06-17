from loader_util.preprocessing import AspectAwarePreprocessor
from imutils import paths
from cv2 import cv2
import numpy as np
import os

sourcePath = r"C:\Users\mhasa\Desktop\temp_cad_threshinv"
targetPath = r"C:\Users\mhasa\Desktop"
folder_name = os.listdir(sourcePath)[0]

# first create the folder in the target folder
destination_folder = f"{targetPath}//{folder_name}"
os.mkdir(destination_folder)

# global variables
imagePaths = list(paths.list_images(sourcePath))
thresh = 210
max_val = 255
target_image_size = 28
aap = AspectAwarePreprocessor(target_image_size, target_image_size)

for path in imagePaths:
    # first read the image
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # get the image name
    image_name = path.split(os.sep)[-1]

    # do inverse thresholding
    th, dst_bin = cv2.threshold(img.copy(),
                                thresh=thresh,
                                maxval=max_val,
                                type=cv2.THRESH_BINARY_INV)

    # find all contours in the image
    contours, hierarchy = cv2.findContours(dst_bin,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)

    # then find bounding box ROI
    new_image = dst_bin.copy()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # extract the ROI
        roi = new_image[y:y + h, x:x + w]

        # resize the ROI
        resizedROI = aap.preprocess(roi)

        # save the ROI
        cv2.imwrite(f"{destination_folder}//{image_name}", roi)
