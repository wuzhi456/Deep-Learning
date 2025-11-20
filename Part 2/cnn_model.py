from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class CNN(nn.Module):

  def __init__(self, n_channels, n_classes):
    """
    Initializes CNN object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
    """
    super(CNN, self).__init__()
    
    # Following VGG-like architecture as mentioned in assignment
    # Block 1: 2 conv layers with 64 filters
    self.conv1_1 = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)
    self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    
    # Block 2: 2 conv layers with 128 filters
    self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    
    # Block 3: 3 conv layers with 256 filters
    self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
    self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
    self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
    self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
    
    # Block 4: 3 conv layers with 512 filters
    self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
    self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
    
    # Block 5: 3 conv layers with 512 filters
    self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
    
    # Fully connected layers
    # After 5 pooling layers, 32x32 image becomes 1x1
    self.fc1 = nn.Linear(512 * 1 * 1, 512)
    self.fc2 = nn.Linear(512, n_classes)
    
    # Activation and dropout
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    """
    Performs forward pass of the input.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    """
    # Block 1
    out = self.relu(self.conv1_1(x))
    out = self.relu(self.conv1_2(out))
    out = self.pool1(out)
    
    # Block 2
    out = self.relu(self.conv2_1(out))
    out = self.relu(self.conv2_2(out))
    out = self.pool2(out)
    
    # Block 3
    out = self.relu(self.conv3_1(out))
    out = self.relu(self.conv3_2(out))
    out = self.relu(self.conv3_3(out))
    out = self.pool3(out)
    
    # Block 4
    out = self.relu(self.conv4_1(out))
    out = self.relu(self.conv4_2(out))
    out = self.relu(self.conv4_3(out))
    out = self.pool4(out)
    
    # Block 5
    out = self.relu(self.conv5_1(out))
    out = self.relu(self.conv5_2(out))
    out = self.relu(self.conv5_3(out))
    out = self.pool5(out)
    
    # Flatten
    out = out.view(out.size(0), -1)
    
    # Fully connected layers
    out = self.dropout(self.relu(self.fc1(out)))
    out = self.fc2(out)
    
    return out
