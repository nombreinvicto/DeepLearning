import cv2
import numpy as np

# declare list to store points for rectangle
top_left_corners = []
bottom_right_corners = []


# resize function
def resize(image: np.ndarray, target_size=(400, 400)):
    return cv2.resize(image, dsize=target_size)


# define the call back function
def draw_rectangle(event, x, y, flags, user_data):
    global top_left_corners, bottom_right_corners, image
    # action to be taken when left mouse button is pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        # clear screen on mouse left button
        image = resize(clear_screen_image.copy(), target_size=(600, 600))

        top_left_corners = [(x, y)]

        # mark the top-left
        cv2.circle(image, top_left_corners[0],
                   1, (255, 255, 0), 2, cv2.LINE_AA)

    elif event == cv2.EVENT_LBUTTONUP:
        bottom_right_corners = [(x, y)]

        # draw the rectangle
        cv2.rectangle(image,
                      top_left_corners[0],
                      bottom_right_corners[0],
                      (0, 255, 0),
                      2, cv2.LINE_AA)

        # now get the cropped image
        h = bottom_right_corners[0][1] - top_left_corners[0][1]
        w = bottom_right_corners[0][0] - top_left_corners[0][0]

        print("h: ", h)
        print("w: ", w)

        cropped = image[top_left_corners[0][1]: top_left_corners[0][1] + h,
                  top_left_corners[0][0]: top_left_corners[0][0] + w]

        cv2.imwrite('face.jpg', cropped)


# read the image
image = cv2.imread('sample.jpg')  # type: np.ndarray

# raise error if image doesnt exist
if image is None:
    raise ValueError('Image doesnt Exist. Try Again....')

# create a dummy to clear the screen when needed
clear_screen_image = image.copy()

# create a named window
cv2.namedWindow('Output')

# highgui function called when mouse event occurs in this window
cv2.setMouseCallback('Output', draw_rectangle)

k = 0

# loop until escape character is pressed
while k != 27:
    image = resize(image, target_size=(600, 600))
    cv2.imshow('Output', image)
    cv2.putText(image, '''Choose top-left corner of rectangle and drag''',
                (10, 570), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    k = cv2.waitKey(20)
    if k == 99:  # if 'c' is pressed to clear the screen
        image = clear_screen_image.copy()
cv2.destroyAllWindows()
