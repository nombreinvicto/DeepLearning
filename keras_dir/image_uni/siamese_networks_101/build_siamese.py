from tensorflow.keras.datasets import mnist
from imutils import build_montages
import numpy as np
import cv2


# %% ##################################################################
def make_pairs(images: np.ndarray,
               labels: np.ndarray):
    # to hold (image, image) pairs
    pair_images = []

    # to hold corr labels (same=1 or not=0)
    pair_labels = []

    # get unique no of classes
    num_classes = len(np.unique(labels))

    # idx is a list of np arrays, each array
    # containing indices of where labels == i, where i starts from 0
    idx = [np.where(labels == i)[0] for i in range(0, num_classes)]

    # loop over the array of images supplied
    print(f"[INFO] looping over: {len(images)} times......")
    for idxa in range(len(images)):
        current_image = images[idxa]
        current_label = labels[idxa]

        # randomly pick an image belonging to the same class label
        idxb = np.random.choice(idx[current_label])
        pos_image = images[idxb]

        pair_images.append([current_image, pos_image])
        pair_labels.append([1])

        # randomly choose a negative image for the current image
        neg_idx = np.where(labels != current_label)[0]
        neg_image = images[np.random.choice(neg_idx)]

        pair_images.append([current_image, neg_image])
        pair_labels.append([0])

    # return the pairs after iterating thru all images
    return np.array(pair_images), np.array(pair_labels)


# %% ##################################################################

print(f"[INFO] loading mnist dataset......")
# mnist has 60K train and 10K test images
(trainx, trainy), (testx, testy) = mnist.load_data()

# build +ve and -ve image pairs, so we expect
# 120K and 20K pairs respectively coz for each
# image we have pos and neg pairs
pair_train, label_train = make_pairs(images=trainx, labels=trainy)
pair_test, label_test = make_pairs(images=testx, labels=testy)
# %% ##################################################################
# init list for montages
images = []
# # randomly select 49 indices and iterate using those indices
for i in np.random.choice(a=np.arange(0, len(pair_train)),
                          size=(49,)):
    # grab the current image pair and label
    image_a = pair_train[i][0]
    image_b = pair_train[i][1]
    label = label_train[i]

    # to make it easier to visualize the pairs and their positive or
    # negative annotations, we're going to "pad" the pair with four
    # pixels along the top, bottom, and right borders, respectively
    # remeber mnist images are 28 x 28
    output = np.zeros((36, 60), dtype="uint8")
    pair = np.hstack([image_a, image_b])
    output[4:32, 0:56] = pair

    # set the text label for the pair along with what color we are
    # going to draw the pair in (green for a "positive" pair and
    # red for a "negative" pair)
    text = "neg" if label[0] == 0 else "pos"
    # BGR
    color = (0, 0, 255) if label[0] == 0 else (0, 255, 0)

    # create a 3-channel RGB image from the grayscale pair, resize
    # it from 60x36 to 96x51 (so we can better see it), and then
    # draw what type of pair it is on the image
    vis = cv2.merge([output] * 3)
    vis = cv2.resize(vis, (96, 51), interpolation=cv2.INTER_LINEAR)
    cv2.putText(vis, text, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    # add the pair visualization to our list of output images
    images.append(vis)
# %% ##################################################################
if __name__ == '__main__':
    # construct the montage for the images
    # montga shape is how many images u want across x and y axes
    # 7 x 7 = 49 images we selected
    montage = build_montages(images, image_shape=(96, 51),
                             montage_shape=(7, 7))[0]
    # show the output montage
    cv2.imshow("Siamese Image Pairs", montage)
    cv2.waitKey(0)
# %% ##################################################################
