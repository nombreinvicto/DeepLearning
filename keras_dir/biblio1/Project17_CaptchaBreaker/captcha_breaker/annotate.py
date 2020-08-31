# import the necessary packages
from imutils import paths
import imutils
import cv2
import os

# construct the argparser
args = {
    'input': 'downloads',
    'annot': 'dataset'
}

# %% grab the image paths then initialise the dict of characters counts
imagePaths = list(paths.list_images(args['input']))
counts = {}

for i, imagePath in enumerate(imagePaths):
    # display an update to the user
    print(imagePath)
    print(f'[INFO] processing image {i + 1}/{len(imagePaths)}')

    try:
        # load the image and convert to grayscale then oad the image
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

        # threshold the image to reveal the digits
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV |
                               cv2.THRESH_OTSU)[1]

        # find contours in the image, keeping only four largest
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]

        # loop over the contours
        for c in cnts:
            # compute the bounding box for the contour then extract the digit
            x, y, w, h = cv2.boundingRect(c)
            roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]

            # display the character making it large enuf for us to see then
            # wait for key press
            cv2.imshow("ROI", imutils.resize(roi, width=28))
            key = cv2.waitKey(0)

            # if key '`' is pressed, then ignore the character
            if key == ord("`"):
                print('[INFO] ignoring character......')
                continue

            # grab the key that was pressed and construct the path
            key = chr(key).upper()
            dirPath = os.path.sep.join([args['annot'], key])

            # if the output directory does not exist , create it
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)

            # write the labelled character to file
            count = counts.get(key, 1)
            p = os.path.sep.join([dirPath, f"{str(count).zfill(6)}.png"])
            cv2.imwrite(p, roi)

            # increment the count for the current key
            counts[key] = count + 1


    except KeyboardInterrupt:
        print('[INFO] manually leaving script')
        break

    except Exception as msg:
        print(msg)
        print('[INFO] skipping image')
