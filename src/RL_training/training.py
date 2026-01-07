"""
DQN Agent for Coffee Roaster Control

This module implements a Deep Q-Network (DQN) agent for learning optimal
coffee roasting strategies using the Aillio Dummy simulator.
"""

import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add parent directory to sys.path to allow importing artisanlib
sys.path.append(str(Path(__file__).resolve().parent.parent))

from artisanlib.aillio_dummy import AillioDummy as Roaster


# =============================================================================
# Neural Network Architecture
# =============================================================================

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


# =============================================================================
# Replay Buffer
# =============================================================================

class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    
    Stores (state, action, reward, next_state, done) tuples and provides
    random sampling for training stability.
    """
    
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
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


# =============================================================================
# DQN Agent
# =============================================================================

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
        device: str = None
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


# =============================================================================
# Training Loop
# =============================================================================

class SuppressOutput:
    """Context manager to suppress stdout."""
    def __init__(self):
        self._devnull = None
        self._original_stdout = None
        
    def __enter__(self):
        import os
        self._original_stdout = sys.stdout
        self._devnull = open(os.devnull, 'w')
        sys.stdout = self._devnull
        return self
    
    def __exit__(self, *args):
        sys.stdout = self._original_stdout
        if self._devnull:
            self._devnull.close()


def create_roaster(debug: bool = False) -> Roaster:
    """Create a roaster instance with optional debug output."""
    if debug:
        roaster = Roaster()
    else:
        with SuppressOutput():
            roaster = Roaster()
    roaster.AILLIO_DEBUG = debug
    return roaster


def train(
    num_episodes: int = 500,
    max_steps: int = 1_200,
    render_interval: int = 50,
    save_interval: int = 100,
    save_path: str = "dqn_roaster.pt",
    verbose: bool = False
) -> List[float]:
    """
    Train the DQN agent on the coffee roaster simulator.
    
    Args:
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode (~20 min at 1s timestep)
        render_interval: Episodes between progress prints
        save_interval: Episodes between model saves
        save_path: Path to save model checkpoints
        verbose: Whether to show roaster debug output
        
    Returns:
        List of total rewards per episode
    """
    # Suppress roaster debug output during training
    import logging
    logging.getLogger('artisanlib.aillio_dummy').setLevel(logging.WARNING)
    
    agent = DQNAgent(
        state_size=4,
        action_size=3,
        hidden_size=128,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        target_update_freq=100
    )
    
    episode_rewards = []
    best_reward = float('-inf')
    
    print(f"Training DQN Agent on {agent.device}")
    print(f"Episodes: {num_episodes}, Max Steps: {max_steps}")
    print("-" * 60)
    
    for episode in range(num_episodes):
        roaster = create_roaster(debug=verbose)
        state = np.array(roaster.get_state_vector(), dtype=np.float32)
        total_reward = 0.0
        total_loss = 0.0
        steps = 0
        
        for step in range(max_steps):
            # Select and perform action
            action_idx = agent.select_action(state)
            action_value = agent.get_action_value(action_idx)
            
            next_state_list, reward, done = roaster.tick(action_value)
            next_state = np.array(next_state_list, dtype=np.float32)
            
            # Store transition and train
            agent.store_transition(state, action_idx, reward, next_state, done)
            loss = agent.train_step()
            
            total_reward += reward
            total_loss += loss
            state = next_state
            steps += 1
            
            if done:
                break
        
        # Decay exploration rate
        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        
        # Update best reward and save
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(save_path.replace('.pt', '_best.pt'))
        
        # Periodic logging
        if episode % render_interval == 0:
            avg_loss = total_loss / max(steps, 1)
            print(
                f"Episode {episode:4d} | "
                f"Reward: {total_reward:8.2f} | "
                f"Best: {best_reward:8.2f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Steps: {steps:4d} | "
                f"Avg Loss: {avg_loss:.4f}"
            )
        
        # Periodic model save
        if episode > 0 and episode % save_interval == 0:
            agent.save(save_path)
    
    # Final save
    agent.save(save_path)
    print("-" * 60)
    print(f"Training complete! Best reward: {best_reward:.2f}")
    
    return episode_rewards


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Training configuration
    NUM_EPISODES = 500
    MAX_STEPS = 1_200  # ~20 minutes at 1s timestep
    
    rewards = train(
        num_episodes=NUM_EPISODES,
        max_steps=MAX_STEPS,
        render_interval=10,
        save_interval=50,
        save_path="./RL_training/dqn_roaster.pt"
    )