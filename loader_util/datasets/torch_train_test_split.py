from imutils import paths
import numpy as np


def train_test_split_paths(dir, test_size=0.3):
    all_image_paths = paths.list_images(dir)
    all_image_paths = np.random.permutation(list(all_image_paths))

    test_index = round(test_size * len(all_image_paths))

    train_paths = all_image_paths[test_index:]
    test_paths = all_image_paths[:test_index]

    return list(train_paths), list(test_paths)
