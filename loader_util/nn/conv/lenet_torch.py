from torch import nn


class LeNet(nn.Module):
    def __init__(self, width, height, depth, classes):
        super(LeNet, self).__init__()

        self.width = width
        self.height = height

        # CONV
        self.cnv1 = nn.Conv2d(in_channels=depth,
                              out_channels=20,
                              kernel_size=5,
                              padding=1,
                              stride=1)
        self.Activation = nn.ReLU()
        self.BatchNorm = nn.BatchNorm2d()


