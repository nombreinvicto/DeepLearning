import os
import random
import hashlib

# %%
dir = r"C:\Users\mhasa\Google Drive\Tutorial " \
      r"Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets" \
      r"\cellImages4\MG63"
dirNamePad = dir + r"\\"

for i, oldFileName in enumerate(os.listdir(dir)):
    print(i + 1, oldFileName)
    oldFileName = dirNamePad + oldFileName
    newFileName = dirNamePad + str(i + 1)
    os.rename(oldFileName, newFileName)

# %%

type_dir_path = dir + r"\\"

cell_dir_path = type_dir_path
for _, cell_dir in enumerate(os.listdir(type_dir_path)):
    for cellImageFile in os.listdir(type_dir_path + cell_dir):
        newRandNumber = random.randint(1, 1000_000_000)
        newName = hashlib.sha256((str(newRandNumber))
                                 .encode('utf8')).hexdigest()

        oldNameOfImage = type_dir_path + cell_dir + '\\' + cellImageFile
        newNameOfImage = type_dir_path + cell_dir + '\\' + newName[
                                                           :10] + '.jpg'

        os.rename(oldNameOfImage, newNameOfImage)
# %%
