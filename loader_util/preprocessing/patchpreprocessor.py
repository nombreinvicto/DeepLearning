from sklearn.feature_extraction.image import extract_patches_2d


class PatchPreprocessor:
    def __init__(self, width, height):
        # store the target width and height of the image
        self.width = width
        self.height = height

    def preprocess(self, image):
        # extract random crop from the image with target width and height
        random_crops = extract_patches_2d(image,
                                          patch_size=(self.height, self.width),
                                          max_patches=1)

        return random_crops[0]
