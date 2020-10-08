from cv2 import cv2
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

thresh = 180
max_val = 255
dir = r"C:\Users\mhasa\Google Drive\Tutorial Corner\PYTH\PyCharm Projects\DeepLearningCV\research\cad_repo_project\bracket.png"

raw_img = cv2.imread(dir)
img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

th, dst_bin = cv2.threshold(img.copy(),
                            thresh=thresh,
                            maxval=max_val,
                            type=cv2.THRESH_BINARY_INV)
print(dst_bin.shape)

# find all contours in the image
contours, hierarchy = cv2.findContours(dst_bin,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)

# # then find bounding box ROI
# new_image = dst_bin.copy()
# print(new_image.shape)

# print(contours)

cv2.drawContours(raw_img, contours, -1, (0, 255, 0), 1)

# cv2.imshow("out", new_image)
plt.imshow(raw_img[:, :, ::-1])
#plt.show()
#%%

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    print(x, y, w, h)

    # extract the ROI
    pad = 0
    print("gone")
    # roi = new_image[y - pad:y + h + pad, x - pad:x + w + pad]

#%%
pad = 10
cv2.rectangle(raw_img, (x-pad, y-pad), (x+w+pad, y+h+pad), (255, 0, 0), 1)
plt.imshow(raw_img[:, :, ::-1])
plt.show()
#%%

roi = raw_img[y - pad:y + h + pad, x - pad:x + w + pad]
plt.imshow(roi[:, :, ::-1])
plt.show()