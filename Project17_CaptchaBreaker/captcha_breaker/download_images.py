# import the necessary packages
import requests
import time
import os

# argument dict
args = {
    'output': 'downloads',
    'num_images': 500
}

# %% Initialise global variables

url = "https://www.e-zpassny.com/vector/jcaptcha.do"
total = 0

# loop over the no of images to download
for i in range(args['num_images']):
    try:
        r = requests.get(url=url, timeout=60)

        # save the image to disk
        p = os.path.sep.join([args['output'], f"{str(total).zfill(5)}.jpg"])
        f = open(p, "wb")
        f.write(r.raw)
        print(r.raw)
        f.close()

        # update counter
        print(f'[INFO] downloaded: {p}')
        total += 1

    except Exception as msg:
        print(msg)
        print('[INFO] Error in downloading image ....')

    # delay for server
    time.sleep(0.1)
