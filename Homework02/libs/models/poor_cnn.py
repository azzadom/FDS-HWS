import torch.nn as nn
import torch

class PoorPerformingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        ##############################
        ###     CHANGE THIS CODE   ###
        ##############################  
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)


        # Changed input channels from 5 to 4 to match conv1 output
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(8 * 4 * 4, 28)

    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x)))
        x = self.pool(self.relu2(self.conv2(x)))
        # Fixed the flatten operation to match actual dimensions
        x = x.view(-1, 8 * 4 * 4)
        x = self.fc1(x)
        return x