# do the necessary imports

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


# find the cad part types
cwd = os.getcwd()
cad_dir = f"{cwd}//static//cad_repo"
part_types = str(os.listdir(cad_dir))
