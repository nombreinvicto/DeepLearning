from loader_util.io import HDF5DatasetGenerator

test_path = r"C:\Users\mhasa\Google Drive\Tutorial Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets\all_cats_dogs\hdf5\test.hdf5"
# reinit the test set generator for crop preprocessing
testgen = HDF5DatasetGenerator(test_path,
                               batchSize=64,
                               classes=2,)

print(f"[INFO] total images = {testgen.numImages}......")

i = 0
for images, labels in testgen.generator():
    print(i, images.shape)
    print(images[0].mean())
    i += 1
    print("=" * 50)