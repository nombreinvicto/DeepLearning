# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes
import imutils
import cv2

sns.set()
# %%
args = {
    'cascade': 'haarcascade_frontalface_default.xml',
    'model': 'lenet_smiles.hdf5',
    'video': ''
}
# %%
# load the face detector and smile detector CNN
detector = cv2.CascadeClassifier(args['cascade'])
model = load_model(args['model'])

# if videopath was not supplied, grab reference to webcam
if not args.get('video', False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args['video'])
# %%

# keep looping
# grab current frame
while True:
    grabbed, frame = camera.read()

    # if we are viewing video and we did not grab a frame, then we have
    # reached end of video
    if args.get('video') and not grabbed:
        break

    # resize the frame, convert to grayscale and then clone the frame for
    # later use
    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameClone = frame.copy()

    # detect faces in the input frame then clone the frame so that we can draw
    rects = detector.detectMultiScale(gray,
                                      scaleFactor=1.1,
                                      minNeighbors=5,
                                      minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    # loop over the face bounding boxes
    for fx, fy, fw, fh in rects:
        # extract the ROI of the face from grayscale image
        # resize it to a fixed 28x28 pixel and then prepare for classification
        roi = gray[fy:fy + fh, fx:fx + fw]
        roi = cv2.resize(roi, dsize=(28, 28))
        roi = roi.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # pass roi thru network
        notSmiling, smiling = model.predict(roi)[0]
        print(notSmiling, smiling)
        label = "Smiling" if smiling > notSmiling else "Not Smiling"

        # display the label and bounding box rectabgle on the output frame
        cv2.putText(frameClone, label, (fx, fy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2)

    # show our ditected faces along with labels
    cv2.imshow('Face', frameClone)

    # if q is pressed stop the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
