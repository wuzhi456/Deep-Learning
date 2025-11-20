from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import torch
import torch.nn as nn
from cnn_model import CNN

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'
DATA_DIR_DEFAULT = './data'

FLAGS = None

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        targets: 1D int array of size [number_of_data_samples] with ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    # Get predicted class (argmax over class dimension)
    pred_classes = torch.argmax(predictions, dim=1)
    # Calculate accuracy
    correct = (pred_classes == targets).float()
    acc = correct.mean().item()
    return acc

def train():
    """
    Performs training and evaluation of CNN model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # Note: This is a skeleton. In a real implementation, you would need to:
    # 1. Load CIFAR10 dataset using torchvision.datasets.CIFAR10
    # 2. Create data loaders for train and test
    # 3. Initialize the CNN model
    # 4. Set up Adam optimizer and CrossEntropy loss
    # 5. Train the model with mini-batch gradient descent
    # 6. Evaluate periodically on test set
    
    print("Training configuration:")
    print(f"  Learning rate: {FLAGS.learning_rate}")
    print(f"  Batch size: {FLAGS.batch_size}")
    print(f"  Max steps: {FLAGS.max_steps}")
    print(f"  Eval frequency: {FLAGS.eval_freq}")
    print(f"  Data directory: {FLAGS.data_dir}")
    print("\nNote: This is a skeleton implementation.")
    print("To use this, you need to load CIFAR10 data in your notebook or script.")

def main():
    """
    Main function
    """
    train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()