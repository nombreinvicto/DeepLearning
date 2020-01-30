from torch import nn


class ShallowNetTorch(nn.Module):
    def __init__(self, width, height, depth, classes):
        super(ShallowNetTorch, self).__init__()

        self.width = width
        self.height = height

        # first and only conv
        self.conv1 = nn.Conv2d(in_channels=depth, out_channels=32,
                               kernel_size=3, stride=1, padding=1)
        # Relu Activation
        self.activation = nn.ReLU()

        # linear layer (32*32*32 -> classes)
        self.fc1 = nn.Linear(self.width * self.height * 32, classes)

    def forward(self, x):
        # add sequence of convolutions
        x = self.activation(self.conv1(x))

        # flatten the activations
        x = x.view(-1, self.width * self.height * 32)

        # pass thru last activations
        x = self.fc1(x)

        return x
