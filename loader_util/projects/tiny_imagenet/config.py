import os

train_images_path = r"C:\Users\mhasa\Downloads\tiny-imagenet-200\train"
val_images_path = r"C:\Users\mhasa\Downloads\tiny-imagenet-200\val\images"

val_mappings_file = r"C:\Users\mhasa\Downloads\tiny-imagenet-200\val\val_annotations.txt"
wordnet_ids_file = r"C:\Users\mhasa\Downloads\tiny-imagenet-200\wnids.txt"
wordnet_map_file = r"C:\Users\mhasa\Downloads\tiny-imagenet-200\words.txt"

num_classes = 200
num_test_images = 50 * num_classes

train_hdf5_path = r"C:\Users\mhasa\Downloads\tiny-imagenet-200\hdf5\train.hdf5"
val_hdf5_path = r"C:\Users\mhasa\Downloads\tiny-imagenet-200\hdf5\val.hdf5"
test_hdf5_path = r"C:\Users\mhasa\Downloads\tiny-imagenet-200\hdf5\test.hdf5"
rgb_mean_path = r"C:\Users\mhasa\Downloads\tiny-imagenet-200\rgb_mean.json"

output_path = r"/content/drive/MyDrive/Colab Notebooks/pyimagesearch/bibilio2/chapter11_minigooglenet/output/tinyimagenet"
model_path = os.path.sep.join([output_path, "epoch_70_model.hdf5"])
fig_path = os.path.sep.join([output_path, "deepergooglenet_on_tinyimagenet.png"])
json_path = os.path.sep.join([output_path, "deepergooglenet_on_tinyimagenet.json"])
