# import the necessary package
import os

# define the TFRecord filenames
TFRECORD_SINGLE_FNAME = "single_data.tfrecord"
TFRECORD_EXAMPLE_FNAME = "example_data.tfrecord"

# define the image url, image file name, and the class for the image
IMAGE_FNAME = "dog.jpg"
IMAGE_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg"
IMAGE_CLASS = "dog"

# define the output path and the image path to save
OUTPUT_PATH = "output"
RESIZED_IMAGE_PATH = os.path.join(OUTPUT_PATH, "resized_image.png")
DESERIALIZED_IMAGE_PATH = os.path.join(OUTPUT_PATH,
	"deserialized_image.png")