from sklearn.feature_extraction.image import extract_patches_2d


class PatchPreprocessor:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def preprocess(self, image):
        # return the single patch
        return extract_patches_2d(image,
                                  patch_size=(self.height, self.width),
                                  max_patches=1)[0]
