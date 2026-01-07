"""
Experience Replay Buffer for DQN Training
==========================================

## Why Do We Need a Replay Buffer?

In traditional supervised learning, we assume training data is independent
and identically distributed (i.i.d.). But in RL, consecutive experiences
are highly correlated - if you're at 150°C now, you'll probably be at ~151°C
next second. Training on correlated data causes the neural network to:
1. Overfit to recent experiences
2. Forget older experiences (catastrophic forgetting)
3. Become unstable during training

## How Experience Replay Solves This:

The replay buffer stores thousands of past experiences (transitions) and
samples RANDOM mini-batches for training. This:
1. Breaks the correlation between consecutive samples
2. Allows reusing rare but important experiences multiple times
3. Smooths out the learning process

## What is a "Transition"?

A transition is one step of interaction with the environment:
    (state, action, reward, next_state, done)

Example in coffee roasting:
    state = [bean_temp=150, ror=8.5, heater=6, time=180]
    action = 2 (increase heater)
    reward = 5.0 (good RoR, staying on target)
    next_state = [bean_temp=151.2, ror=8.3, heater=7, time=181]
    done = False (roast not finished)

## Technical Implementation:

This uses a "deque" (double-ended queue) with a maximum size. When full,
the oldest experiences are automatically removed to make room for new ones.
This is called a "circular buffer" or "ring buffer" pattern.
"""

import random
from collections import deque
from typing import Tuple

import numpy as np


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.

    This class implements a circular buffer that stores past experiences
    and provides random sampling for stable neural network training.

    Attributes:
        buffer: A deque (double-ended queue) storing transition tuples
        capacity: Maximum number of transitions to store

    Example Usage:
        >>> buffer = ReplayBuffer(capacity=10000)
        >>> buffer.push(state, action, reward, next_state, done)
        >>> batch = buffer.sample(batch_size=32)
    """

    def __init__(self, capacity: int = 100_000):
        """
        Initialize the replay buffer.

        Args:
            capacity: Maximum number of transitions to store. Once full,
                      oldest transitions are automatically removed.
                      Default is 100,000 which is typical for DQN.

        Note:
            Larger buffers = more diverse samples but more memory usage.
            Smaller buffers = faster training but may forget useful experiences.
            100k is a good balance for most applications.
        """
        # deque with maxlen automatically removes oldest items when full
        # This is much more efficient than manually managing a list
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Store a single transition (experience) in the buffer.

        This is called after every step the agent takes in the environment.
        The transition captures everything needed to learn from this experience.

        Args:
            state: The state BEFORE taking the action.
                   Shape: (state_size,) e.g., [bean_temp, ror, heater, time]
            action: The action that was taken (0, 1, or 2 for our roaster)
            reward: The reward received after taking this action
            next_state: The state AFTER taking the action
            done: True if this action ended the episode (roast finished)

        The Bellman Equation (core of Q-learning) uses these components:
            Q(state, action) = reward + gamma * max(Q(next_state, all_actions))

        If done=True, there is no next_state to consider, so:
            Q(state, action) = reward
        """
        # Store as a tuple - this is memory efficient
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Randomly sample a batch of transitions for training.

        Random sampling is CRITICAL - it breaks the correlation between
        consecutive experiences that would otherwise destabilize training.

        Args:
            batch_size: Number of transitions to sample (typically 32-128)

        Returns:
            Tuple of numpy arrays: (states, actions, rewards, next_states, dones)
            Each array has shape (batch_size, ...) for efficient batch processing

        Note:
            Only call this when len(buffer) >= batch_size, otherwise
            random.sample will raise an error.
        """
        # random.sample returns a list of random items WITHOUT replacement
        # This means each transition in the batch is unique
        batch = random.sample(self.buffer, batch_size)

        # zip(*batch) is a Python trick to "transpose" the list of tuples
        # [(s1,a1,r1,ns1,d1), (s2,a2,r2,ns2,d2)] -> [(s1,s2), (a1,a2), ...]
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to numpy arrays for efficient tensor conversion later
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self) -> int:
        """
        Return the current number of transitions stored.

        Used to check if we have enough samples to start training:
            if len(buffer) >= batch_size:
                train()
        """
        return len(self.buffer)
