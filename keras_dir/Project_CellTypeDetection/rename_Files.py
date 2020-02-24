import os
import random
import hashlib

dir = r"C:\Users\mhasa\Google Drive\Tutorial " \
      r"Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets" \
      r"\cellImages2" \
      r"\MG63\5"
dirNamePad = dir + r"\\"
p = os.listdir(dir)

# %%
for fileName in os.listdir(dir):
    oldFileName = dirNamePad + fileName

    newRandNumber = random.randint(1, 1000_000_000)
    newName = hashlib.sha256((str(newRandNumber))
                             .encode('utf8')).hexdigest()
    newName = dirNamePad + newName[:10] + '.jpg'
    os.rename(oldFileName, newName)
# %%

