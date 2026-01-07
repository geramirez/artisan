"""
Q-Network: Neural Network for Q-Value Approximation
====================================================

## What is a Q-Value?

In reinforcement learning, Q(s, a) represents the "quality" of taking
action 'a' in state 's'. Specifically, it's the expected total future
reward if you:
1. Take action 'a' in state 's'
2. Then follow the optimal policy forever after

Example in coffee roasting:
    State: bean_temp=180°C, ror=6.0, heater=8, fan=5, time=400s
    
    Q-values for all 9 actions (heater_change, fan_change):
    Q(state, action_0) = 45.2   # (-1, -1) decrease both
    Q(state, action_1) = 48.1   # (-1,  0) decrease heater, keep fan
    Q(state, action_2) = 44.0   # (-1, +1) decrease heater, increase fan
    Q(state, action_3) = 50.5   # ( 0, -1) keep heater, decrease fan
    Q(state, action_4) = 52.8   # ( 0,  0) keep both <- BEST
    Q(state, action_5) = 49.2   # ( 0, +1) keep heater, increase fan
    Q(state, action_6) = 42.1   # (+1, -1) increase heater, decrease fan
    Q(state, action_7) = 38.5   # (+1,  0) increase heater, keep fan
    Q(state, action_8) = 35.0   # (+1, +1) increase both

    The agent would choose action_4 (keep both) since it has the highest Q-value.

## Why Use a Neural Network?

In simple problems, you can store Q-values in a table (one entry per
state-action pair). But coffee roasting has CONTINUOUS states - there
are infinite possible temperatures! A neural network can:
1. Generalize from seen states to unseen states
2. Handle continuous state spaces
3. Learn complex patterns in the data

## Network Architecture:

    Input Layer (5 neurons)
         │
         │  state = [bean_temp, ror, heater_level, fan_level, time]
         ▼
    Hidden Layer 1 (128 neurons) + ReLU activation
         │
         │  Learns low-level features
         ▼
    Hidden Layer 2 (128 neurons) + ReLU activation
         │
         │  Learns higher-level patterns
         ▼
    Output Layer (9 neurons)
         │
         └──► [Q(s,a0), Q(s,a1), ..., Q(s,a8)]
              One Q-value per heater/fan action combination

## What is ReLU?

ReLU (Rectified Linear Unit) is an activation function: f(x) = max(0, x)
It introduces non-linearity, allowing the network to learn complex patterns.
Without activation functions, stacking layers would just be linear algebra.
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Neural network that approximates Q-values for each action given a state.

    This is a simple feedforward network (also called Multi-Layer Perceptron).
    It takes a state vector as input and outputs Q-values for all possible actions.

    The agent uses these Q-values to decide which action to take:
        action = argmax(Q-values)  # Choose action with highest Q-value

    Architecture:
        Input  -> [state_size] neurons (5 for roaster: temp, ror, heater, fan, time)
        Hidden -> [hidden_size] neurons with ReLU
        Hidden -> [hidden_size] neurons with ReLU
        Output -> [action_size] neurons (9 for roaster: 3 heater × 3 fan options)

    Attributes:
        fc1: First fully-connected (linear) layer
        fc2: Second fully-connected layer
        fc3: Output layer (no activation - Q-values can be any real number)

    Example:
        >>> net = QNetwork(state_size=5, action_size=9)
        >>> state = torch.tensor([[150.0, 8.5, 6.0, 4.0, 180.0]])  # batch of 1
        >>> q_values = net(state)  # Returns tensor of shape (1, 9)
        >>> best_action = q_values.argmax(dim=1)  # Index of highest Q-value
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        """
        Initialize the Q-Network.

        Args:
            state_size: Dimension of the state vector (5 for our roaster:
                        bean_temp, ror, heater_level, fan_level, time)
            action_size: Number of possible actions (9 for our roaster:
                         3 heater options × 3 fan options)
            hidden_size: Number of neurons in hidden layers. Larger = more
                         capacity to learn complex patterns, but slower and
                         may overfit. 128 is a good starting point.
        """
        # nn.Module.__init__() sets up PyTorch internals for the network
        super(QNetwork, self).__init__()

        # nn.Linear(in, out) creates a fully-connected layer
        # It learns a weight matrix W and bias b: output = input @ W + b

        # Layer 1: State -> Hidden representation
        # Takes 4D state, projects to 128D hidden space
        self.fc1 = nn.Linear(state_size, hidden_size)

        # Layer 2: Hidden -> Hidden (deeper feature extraction)
        # 128D -> 128D, learns more abstract patterns
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Layer 3: Hidden -> Q-values
        # 128D -> 3D (one Q-value per action)
        # No activation here! Q-values can be any real number.
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute Q-values for given state(s).

        This method defines how data flows through the network.
        PyTorch automatically tracks gradients through these operations
        for backpropagation during training.

        Args:
            x: State tensor of shape (batch_size, state_size)
               Example: [[150.0, 8.5, 6.0, 4.0, 180.0],
                         [160.0, 7.2, 7.0, 5.0, 240.0]]

        Returns:
            Q-value tensor of shape (batch_size, action_size)
            Example: [[45.2, 48.1, 44.0, 50.5, 52.8, 49.2, 42.1, 38.5, 35.0],
                      [47.0, 49.3, 45.1, 51.2, 53.0, 50.1, 43.2, 39.5, 36.0]]
            Each row contains Q-values for all 9 heater/fan action combinations.

        Note:
            The agent picks actions by taking argmax of these Q-values:
            best_actions = q_values.argmax(dim=1)  # -> [4, 4] (keep both)
        """
        # Layer 1 + ReLU: Learn initial features from state
        # ReLU(x) = max(0, x) - sets negative values to 0
        x = torch.relu(self.fc1(x))

        # Layer 2 + ReLU: Learn deeper/more abstract features
        x = torch.relu(self.fc2(x))

        # Output layer (no activation): Produce Q-value estimates
        # These can be negative (bad action) or positive (good action)
        return self.fc3(x)
