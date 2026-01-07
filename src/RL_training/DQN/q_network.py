"""
Deep Q-Network architecture for approximating Q-values.
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Deep Q-Network architecture for approximating Q-values.

    Architecture:
        - Input: State vector (bean_temp, ror, heater_level, time)
        - Hidden layers: 2 fully connected layers with ReLU activation
        - Output: Q-values for each action
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
