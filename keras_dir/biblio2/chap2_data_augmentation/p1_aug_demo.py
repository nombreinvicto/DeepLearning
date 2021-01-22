from tensorflow.keras.preprocessing.image import \
    ImageDataGenerator, \
    img_to_array, \
    load_img
import numpy as np

# %%

# construct the argument parser
path = r"C:\Users\mhasa\Google Drive\Tutorial Corner\PYTH\PyCharm Projects\DeepLearning\keras_dir\biblio2\chap2_data_augmentation\sample_images"
args = {
    'image': f"{path}/car_pic.jpg",
    'output': r'./output'
}

# %%

# load the input image, convert to Numpy array
print(f'[INFO] loading example image.....')
image = load_img(path=args["image"])
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
# %%

# construct image generator
aug = ImageDataGenerator(rotation_range=30,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode='nearest')
total = 0
# %%
print(f'[INFO] generating images.....')
image_gen = aug.flow(image,
                     batch_size=1,
                     save_to_dir=args["output"],
                     save_prefix="augmented__",
                     save_format="JPG")

for img in image_gen:
    total += 1
    print(img.shape)

    if total == 10:
        break
# %%
