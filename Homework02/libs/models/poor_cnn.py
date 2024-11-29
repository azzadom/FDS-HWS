import torch.nn as nn
import torch

class PoorPerformingCNN(nn.Module):
    def __init__(self):
        super(PoorPerformingCNN, self).__init__() 
        ##############################
        ###     CHANGE THIS CODE   ###
        ##############################  
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Change the input channel from 5 to 4 because the output of the first convolutional layer is 4
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        # Change the input size from 8 * 4 * 4 to 8 * 8 * 8 
        # Change the output size from 28 to 10 because CIFAR-10 has 10 classes
        self.fc1 = nn.Linear(8 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x)))
        x = self.pool(self.relu2(self.conv2(x)))
        # Correct the input size of the fully connected layer to match the output of the last convolutional layer
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x