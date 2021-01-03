# import the necessary packages
import numpy as np
import cv2

# initialize the class labels and set the seed of the pseudorandom
# number generator so we can reproduce our results

labels = ['dog', 'cat', 'panda']
np.random.seed(1)

# randomly generate the weight, W and the bias, b matrices
W = np.random.randn(3, 3072)
b = np.random.rand(3)

# load our example image, resize it and then flatten into our
# "feature vector" representation

orig = cv2.imread("beagle.png")  # type: np.ndarray
image = cv2.resize(orig, (32, 32)).flatten()

# compute the output score
scores = W.dot(image) + b

for label, score in zip(labels, scores):
    print(f"[INFO] label:{label}, score:{score:0.2f}")

# draw the label with the highest score on the image as our prediction
cv2.putText(orig, f"Label: {labels[np.argmax(scores)]}", (10, 30),
            cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
cv2.imshow("Image", orig)
cv2.waitKey(0)
