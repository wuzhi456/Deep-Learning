from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import torch
import torch.nn as nn
from pytorch_mlp import MLP

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10

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

def train(model, train_loader, test_loader, n_epochs, learning_rate, eval_freq=10, device='cpu'):
    """
    Performs training and evaluation of MLP model.
    
    Args:
        model: MLP model instance
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        n_epochs: number of training epochs
        learning_rate: learning rate for optimizer
        eval_freq: frequency of evaluation on test set
        device: device to train on ('cpu' or 'cuda')
    
    Returns:
        train_losses: list of training losses
        train_accs: list of training accuracies
        test_accs: list of test accuracies
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    train_losses = []
    train_accs = []
    test_accs = []
    
    print(f"Training MLP for {n_epochs} epochs...")
    print(f"Learning rate: {learning_rate}, Eval frequency: {eval_freq}")
    print()
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        
        # Training
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Flatten input if it's an image (has more than 2 dimensions)
            if len(batch_x.shape) > 2:
                batch_x = batch_x.view(batch_x.size(0), -1)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluation
        if (epoch + 1) % eval_freq == 0:
            model.eval()
            with torch.no_grad():
                # Train accuracy
                train_correct = 0
                train_total = 0
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    # Flatten input if it's an image (has more than 2 dimensions)
                    if len(batch_x.shape) > 2:
                        batch_x = batch_x.view(batch_x.size(0), -1)
                    outputs = model(batch_x)
                    train_correct += (torch.argmax(outputs, dim=1) == batch_y).sum().item()
                    train_total += batch_y.size(0)
                train_acc = train_correct / train_total
                train_accs.append(train_acc)
                
                # Test accuracy
                test_correct = 0
                test_total = 0
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    # Flatten input if it's an image (has more than 2 dimensions)
                    if len(batch_x.shape) > 2:
                        batch_x = batch_x.view(batch_x.size(0), -1)
                    outputs = model(batch_x)
                    test_correct += (torch.argmax(outputs, dim=1) == batch_y).sum().item()
                    test_total += batch_y.size(0)
                test_acc = test_correct / test_total
                test_accs.append(test_acc)
                
                print(f"Epoch [{epoch+1}/{n_epochs}] - Loss: {avg_loss:.4f}, "
                      f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    print("\nTraining completed!")
    return train_losses, train_accs, test_accs


def main():
    """
    Main function
    """
    train()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                          help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()
    main()