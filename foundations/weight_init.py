import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2 / (fan_in + fan_out))
        W = torch.randn(fan_out, fan_in) * std
        W = torch.round(W, decimals=4) 
        W = W.tolist()
        return W

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2 / fan_in)
        W = torch.randn(fan_out, fan_in) * std
        W = torch.round(W, decimals=4) 
        W = W.tolist()
        return W

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Use torch.manual_seed(0) once at the start.
        # Sample all weights in layer order with torch.randn(out_dim, in_dim) * std:
        # first (hidden_dim, input_dim), then (hidden_dim, hidden_dim).
        # Keep weights unrounded; do not call the init helpers, which reset the seed.
        # Then sample x = torch.randn(1, input_dim) without resetting the seed.
        # Apply x = torch.relu(x @ W.T) at every layer, with no biases.
        # Return the std of activations after each layer, rounded to 2 decimals.
        # Use round(x.std().item(), 2) with PyTorch's default sample std.
        # Step 1: Set the random seed
        torch.manual_seed(0)

        # Step 2: Create a list to store all weight matrices
        weights = []

        for i in range(num_layers):

            # First layer uses input_dim as its input size
            # All later layers use hidden_dim
            if i == 0:
                fan_in = input_dim
            else:
                fan_in = hidden_dim

            fan_out = hidden_dim

            # Step 3: Choose the initialization method
            if init_type == "xavier":
                std = math.sqrt(2 / (fan_in + fan_out))

            elif init_type == "kaiming":
                std = math.sqrt(2 / fan_in)

            elif init_type == "random":
                std = 1.0

            else:
                raise ValueError("Unknown initialization type")

            # Generate the weight matrix for this layer
            W = torch.randn(fan_out, fan_in) * std

            # Store it without rounding
            weights.append(W)

        # Step 4: Generate the input
        x = torch.randn(1, input_dim)

        # Step 5: Forward propagation
        activation_stds = []

        for W in weights:

            # Linear transformation followed by ReLU
            x = torch.relu(x @ W.T)

            # Calculate and store activation standard deviation
            activation_stds.append(round(x.std().item(), 2))

        return activation_stds
