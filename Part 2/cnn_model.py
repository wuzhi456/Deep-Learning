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
    
    # Simplified VGG-like architecture with batch normalization for better training
    # Block 1: 2 conv layers with 64 filters
    self.conv1_1 = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)
    self.bn1_1 = nn.BatchNorm2d(64)
    self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
    self.bn1_2 = nn.BatchNorm2d(64)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    
    # Block 2: 2 conv layers with 128 filters
    self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    self.bn2_1 = nn.BatchNorm2d(128)
    self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    self.bn2_2 = nn.BatchNorm2d(128)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    
    # Block 3: 2 conv layers with 256 filters (reduced from 3)
    self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
    self.bn3_1 = nn.BatchNorm2d(256)
    self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
    self.bn3_2 = nn.BatchNorm2d(256)
    self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
    
    # Block 4: 2 conv layers with 512 filters (reduced from 3)
    self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
    self.bn4_1 = nn.BatchNorm2d(512)
    self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    self.bn4_2 = nn.BatchNorm2d(512)
    self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
    
    # Fully connected layers
    # After 4 pooling layers, 32x32 image becomes 2x2
    self.fc1 = nn.Linear(512 * 2 * 2, 512)
    self.bn_fc1 = nn.BatchNorm1d(512)
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
    out = self.relu(self.bn1_1(self.conv1_1(x)))
    out = self.relu(self.bn1_2(self.conv1_2(out)))
    out = self.pool1(out)
    
    # Block 2
    out = self.relu(self.bn2_1(self.conv2_1(out)))
    out = self.relu(self.bn2_2(self.conv2_2(out)))
    out = self.pool2(out)
    
    # Block 3
    out = self.relu(self.bn3_1(self.conv3_1(out)))
    out = self.relu(self.bn3_2(self.conv3_2(out)))
    out = self.pool3(out)
    
    # Block 4
    out = self.relu(self.bn4_1(self.conv4_1(out)))
    out = self.relu(self.bn4_2(self.conv4_2(out)))
    out = self.pool4(out)
    
    # Flatten
    out = out.view(out.size(0), -1)
    
    # Fully connected layers
    out = self.relu(self.bn_fc1(self.fc1(out)))
    out = self.dropout(out)
    out = self.fc2(out)
    
    return out
