from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(VanillaRNN, self).__init__()
        
        # Store dimensions
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        
        # Initialize parameters for RNN following equations:
        # h(t) = tanh(Whx * x(t) + Whh * h(t-1) + bh)
        # o(t) = Wph * h(t) + bo
        # y_tilde(t) = softmax(o(t))
        
        self.Whx = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.01)
        self.Whh = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.bh = nn.Parameter(torch.zeros(hidden_dim))
        
        self.Wph = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.01)
        self.bo = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        """
        Forward pass through the RNN.
        Args:
            x: input tensor of shape (batch_size, seq_length)
        Returns:
            out: output tensor of shape (batch_size, output_dim) for the last timestep
        """
        # Initialize hidden state h(0) to zeros
        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        
        # Process sequence step by step
        for t in range(self.seq_length):
            # Get input at time t and reshape to (batch_size, input_dim)
            x_t = x[:, t].unsqueeze(1)  # (batch_size, 1)
            
            # Compute h(t) = tanh(Whx * x(t) + Whh * h(t-1) + bh)
            h = torch.tanh(torch.mm(x_t, self.Whx) + torch.mm(h, self.Whh) + self.bh)
        
        # Compute output for last timestep
        # o(T) = Wph * h(T) + bo
        out = torch.mm(h, self.Wph) + self.bo
        
        return out
