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

def train(model, train_loader, test_loader, n_epochs, learning_rate, device='cpu'):
    """
    Performs training and evaluation of CNN model.
    
    Args:
        model: CNN model instance
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        n_epochs: number of training epochs
        learning_rate: learning rate for Adam optimizer
        device: device to train on ('cpu' or 'cuda')
    
    Returns:
        train_losses: list of training losses per epoch
        train_accs: list of training accuracies per epoch
        test_losses: list of test losses per epoch
        test_accs: list of test accuracies per epoch
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    print(f"Training CNN for {n_epochs} epochs...")
    print(f"Learning rate: {learning_rate}, Device: {device}")
    print()
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print progress every 500 batches
            if (i + 1) % 500 == 0:
                print(f'  Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Evaluation phase
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_epoch_loss = test_loss / len(test_loader)
        test_epoch_acc = correct / total
        test_losses.append(test_epoch_loss)
        test_accs.append(test_epoch_acc)
        
        print(f'Epoch [{epoch+1}/{n_epochs}]: '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
              f'Test Loss: {test_epoch_loss:.4f}, Test Acc: {test_epoch_acc:.4f}')
    
    print("\nTraining completed!")
    return train_losses, train_accs, test_losses, test_accs

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