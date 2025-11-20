from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        
        # Initialize weights and biases
        self.Wxh = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.01)
        self.Whh = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.Why = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.01)
        self.bh = nn.Parameter(torch.zeros(hidden_dim))
        self.by = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        # Implementation here ...
        # x shape: (batch_size, seq_length)
        batch_size = x.size(0)
        
        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        # Process sequence
        for t in range(self.seq_length):
            # Get input at time t and reshape to (batch_size, input_dim)
            x_t = x[:, t].unsqueeze(1)  # (batch_size, 1)
            
            # RNN cell computation: h_t = tanh(x_t * Wxh + h_{t-1} * Whh + bh)
            h = torch.tanh(torch.matmul(x_t, self.Wxh) + torch.matmul(h, self.Whh) + self.bh)
        
        # Output layer: y = h * Why + by
        out = torch.matmul(h, self.Why) + self.by
        
        return out
        
    # add more methods here if needed
