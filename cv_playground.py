# %%
from loader_util.utils.simple_obj_det import sliding_window, image_pyramid
from cv2 import cv2

image_path = r"C:\Users\mhasa\PycharmProjects\Py_Play\sample_image.JPG"
image = cv2.imread(image_path)

# for x, y, slice_ in sliding_window(image=image, step=8, ws=(100, 100)):
#     cv2.imshow('slice', slice_)
#     cv2.waitKey(0)


for image in image_pyramid(image):
    cv2.imshow('slice', image)
    cv2.waitKey(0)
