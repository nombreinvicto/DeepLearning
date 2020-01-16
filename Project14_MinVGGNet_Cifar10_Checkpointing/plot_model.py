# import the necessary packages
from loader_util.nn.conv import MinVGGNet
from tensorflow.keras.utils import plot_model

# initlaise the model and store viz to disk
model = MinVGGNet.build(width=32, height=32, depth=3, classes=10)
plot_model(model=model, to_file="MinVGGNet.jpg", show_shapes=True)