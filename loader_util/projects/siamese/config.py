import os

# specify shape of inputs to siamese networks
img_shape = (28, 28, 1)

# specify batch_size and epochs
batch_size = 64
num_epochs = 100

# define path to outputs
base_output = "output"
model_path = os.path.sep.join([base_output, "siamese_model"])
plot_path = os.path.sep.join([base_output, "plot.png"])
