"""
Experience replay buffer for storing and sampling transitions.
"""

import random
from collections import deque
from typing import Tuple

import numpy as np


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.

    Stores (state, action, reward, next_state, done) tuples and provides
    random sampling for training stability.
    """

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a random batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self) -> int:
        return len(self.buffer)
