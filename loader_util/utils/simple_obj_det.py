# import the required packages
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import Model
import numpy as np
import imutils


# %%

def sliding_window(image: np.ndarray, step, ws) -> np.ndarray:
    # slide a windows across the image
    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] - ws[0], step):
            # yield the current  window and location of the window(x,y coordinates)
            yield x, y, image[y:y + ws[1], x:x + ws[0]]


def image_pyramid(image: np.ndarray, scale: float = 1.5, minsize: tuple = (224, 224)) -> np.ndarray:
    # yield the original image
    yield image

    # keep looping over the image pyramid
    while True:
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)  # this does aspect aware resizing

        # if resized image is less in dimension across any axis than a min dim then break the pyramid creation
        if image.shape[0] < minsize[1] or image.shape[1] < minsize[0]:
            break

        yield image


def classify_batch(model: Model, batch_rois: np.ndarray, batch_locs: tuple,
                   labels: dict, min_prob=0.5, top=10, dims=(224, 224)):
    # pass batchrois thru model and get preds
    preds = model.predict(batch_rois)
    P = imagenet_utils.decode_predictions(preds, top=top)

    # loop over decoded predictions
    for i in range(len(P)):
        for _, label, prob_score in P[i]:
            # filter out weak predictions
            if prob_score > min_prob:
                # grab the coordinates of the sliding window and construct the bounding box
                px, py = batch_locs[i]
                box = (px, py, px + dims[0], py + dims[1])

                L = labels.get(label, [])
                L.append((box, prob_score))
                labels[label] = L

    return labels
