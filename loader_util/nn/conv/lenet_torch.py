# define the CNN architecture

from torch import nn


class LeNet(nn.Module):

    def forward(self, x):
        # Define forward behavior
        x = self.conv_layer1(x)
        x = self.fc_layers(x)
        return x

    def __init__(self, classes=2):
        super(LeNet, self).__init__()

        # Define layers of a CNN
        # conv_section -1
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(100352, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, classes)
        )
