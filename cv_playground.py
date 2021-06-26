from sklearn.feature_extraction.image import extract_patches_2d
import matplotlib.pyplot as plt
from cv2 import cv2

path = r"C:\Users\mhasa\Google Drive\Tutorial Corner\PYTH\PyCharm Projects\PyPlay\sample_image.JPG"

image = cv2.imread(path)
plt.imshow(image[:, :, ::-1])
plt.show()
# %%

random_crops = extract_patches_2d(image, max_patches=3, patch_size=(100,100))
#%%

