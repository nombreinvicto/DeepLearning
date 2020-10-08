# from loader_util.preprocessing import AspectAwarePreprocessor
from imutils import paths
from cv2 import cv2
import progressbar
import os

sourcePath = r"C:\Users\mhasa\Desktop\retrieval_models"
targetPath = r"C:\Users\mhasa\Desktop\retrieval_model_color_roi_28px"

# global variables
imagePaths = list(paths.list_images(sourcePath))
thresh = 180
max_val = 255
target_image_size = 28
# aap = AspectAwarePreprocessor(target_image_size, target_image_size)

# init the progressbar
widgets = ["Thresholding..... : ", " ", progressbar.Percentage(),
           " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

for i, path in enumerate(imagePaths):
    # first read the image
    print(f'[INFO] reading image: {path}.....')
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # get the image category type and create a folder
    category_folder = path.split(os.path.sep)[-2]

    # create a folder in destination if it doesnt exist
    if category_folder not in os.listdir(targetPath):
        destination_folder = f"{targetPath}//{category_folder}"
        os.mkdir(destination_folder)

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
    # new_image = dst_bin.copy()
    new_image = img.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # extract the ROI
        pad = 10
        roi = new_image[y - pad:y + h + pad, x - pad:x + w + pad]

        # resize the ROI
        # resizedROI = aap.preprocess(roi)
        resizedROI = cv2.resize(roi, (target_image_size, target_image_size))

        # save the ROI
        cv2.imwrite(f"{destination_folder}//{image_name}", resizedROI)

    pbar.update(i)
