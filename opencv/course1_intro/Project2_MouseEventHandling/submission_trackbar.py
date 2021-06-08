import cv2
import numpy as np


# resize function
def resize(image: np.ndarray, target_size=(400, 400)):
    return cv2.resize(image, dsize=target_size)


# putText function
def put_text(image: np.ndarray):
    image_height = image.shape[0]
    cv2.putText(image, 'Hit ESC to close Window',
                (10, image_height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)


# define the callbacks
def scale_factor_change_cb(*args):
    global default_scale_factor, \
        default_scale_type, \
        image, \
        init_image, \
        current_scale_type

    # get the scale factor from trackbar
    scaleFactor = 1 + ((-1) ** current_scale_type) * (args[0] / 100.0)
    print("Scale Factor: ", scaleFactor)

    # Perform check if scaleFactor is zero
    if scaleFactor == 0:
        scaleFactor = 1

    # Resize the image
    scaledImage = cv2.resize(init_image,
                             None,
                             fx=scaleFactor,
                             fy=scaleFactor,
                             interpolation=cv2.INTER_LINEAR)
    image = scaledImage.copy()
    print("Current Shape: ", image.shape[:2])


def scale_type_change_cb(*args):
    global current_scale_type, init_image
    current_scale_type = args[0]

    # resize the target image to track image size history
    current_image_shape = image.shape[:2]
    # init_image = resize(init_image, target_size=current_image_shape)


# initialise global variables
window_name = 'Output'

# scale factor change trackbar
scale_factor_tb_name = 'Scale'
max_scale_factor = 100
default_scale_factor = 0

# scale up/down change trackbar
scale_type_tb_name = f'Scale Type'
max_scale_type = 1
default_scale_type = 0
current_scale_type = default_scale_type

# load the image to be shown
image = cv2.imread('sample.jpg')
image = resize(image, target_size=(600, 600))
init_image = image.copy()

# create a window to display results
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

# create the trackbar objects
# 1. the scale factor change trackbar
cv2.createTrackbar(scale_factor_tb_name,
                   window_name,
                   default_scale_factor,
                   max_scale_factor,
                   scale_factor_change_cb)
# 2. the scale type change trackbar
cv2.createTrackbar(scale_type_tb_name,
                   window_name,
                   default_scale_type,
                   max_scale_type,
                   scale_type_change_cb)

k = 0
# loop until escape character is pressed
while k != 27:
    cv2.imshow('Output', image)
    put_text(image)
    k = cv2.waitKey(25)
cv2.destroyAllWindows()
