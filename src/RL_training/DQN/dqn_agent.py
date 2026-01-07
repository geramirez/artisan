"""
Deep Q-Network Agent for coffee roaster control.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from RL_training.DQN.q_network import QNetwork
from RL_training.DQN.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Network Agent for coffee roaster control.

    Implements:
        - Double DQN with target network
        - Experience replay
        - Epsilon-greedy exploration

    Actions:
        0: Decrease heater (-1)
        1: Keep heater (0)
        2: Increase heater (+1)
    """

    # Action mapping: index -> heater change
    ACTION_MAP = {0: -1, 1: 0, 2: 1}

    def __init__(
        self,
        state_size: int = 4,
        action_size: int = 3,
        hidden_size: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: str | None = None
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Networks
        self.policy_net = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and replay buffer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training step counter
        self.steps_done = 0

    def select_action(self, state: np.ndarray) -> int:
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: Current state observation

        Returns:
            Action index (0, 1, or 2)
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def get_action_value(self, action_idx: int) -> int:
        """Convert action index to heater change value."""
        return self.ACTION_MAP[action_idx]

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> float:
        """
        Perform one training step using a batch from replay buffer.

        Returns:
            Loss value for logging
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute current Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values (Double DQN)
        with torch.no_grad():
            # Select best actions using policy network
            next_actions = self.policy_net(next_states).argmax(dim=1)
            # Evaluate using target network
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Compute loss and update
        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network periodically
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self) -> None:
        """Copy weights from policy network to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str) -> None:
        """Save model weights and training state."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load model weights and training state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        print(f"Model loaded from {filepath}")
