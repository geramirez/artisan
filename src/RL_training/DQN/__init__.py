"""
DQN (Deep Q-Network) Module for Coffee Roaster Control
======================================================

This module implements a Deep Q-Network reinforcement learning agent
designed to learn optimal coffee roasting strategies.

## What is Reinforcement Learning (RL)?

RL is a type of machine learning where an "agent" learns to make decisions
by interacting with an "environment". The agent:
1. Observes the current STATE of the environment
2. Takes an ACTION based on that state
3. Receives a REWARD signal (positive or negative)
4. Observes the new state
5. Learns from this experience to maximize future rewards

Think of it like training a dog: good actions get treats (positive rewards),
bad actions get scolded (negative rewards), and over time the dog learns
what behaviors lead to treats.

## What is DQN?

DQN (Deep Q-Network) is a specific RL algorithm that uses a neural network
to approximate the "Q-function" - a function that predicts how good each
action is in a given state.

Key components:
- **Q-Network**: Neural network that estimates Q-values for each action
- **Replay Buffer**: Memory that stores past experiences for training
- **Target Network**: A stable copy of the Q-network used for computing targets

## In Coffee Roasting Context:

- **State** (5 elements): Bean temperature, rate of rise (RoR), heater level,
                          fan level, time
- **Actions** (9 options): Combinations of heater and fan adjustments:
    - Heater: decrease (-1), keep (0), increase (+1)
    - Fan: decrease (-1), keep (0), increase (+1)
    - Total: 3 × 3 = 9 action combinations
- **Rewards**: Based on achieving good roast profiles (smooth RoR curves,
              hitting first crack at right temperature, etc.)
- **Goal**: Learn to control heater AND fan together to produce consistent,
            quality roasts

Module Components:
    DQNAgent: The main agent that learns and makes decisions
    QNetwork: The neural network architecture
    ReplayBuffer: Memory for storing and sampling experiences
"""

from RL_training.DQN.dqn_agent import DQNAgent
from RL_training.DQN.q_network import QNetwork
from RL_training.DQN.replay_buffer import ReplayBuffer

__all__ = ["DQNAgent", "QNetwork", "ReplayBuffer"]
