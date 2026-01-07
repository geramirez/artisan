"""
DQN Agent: The Learning Controller
===================================

## Overview

This module implements the DQN (Deep Q-Network) agent that learns to control
the coffee roaster. The agent is the "brain" that:
1. Observes the roaster state (temperatures, time, etc.)
2. Decides what action to take (adjust heater)
3. Learns from the results to make better decisions

## Key Concepts Implemented:

### 1. Double DQN with Target Network

Standard DQN tends to overestimate Q-values because it uses the same network
to both SELECT and EVALUATE actions. Double DQN fixes this:
- Policy Network: Selects which action is best
- Target Network: Evaluates how good that action actually is

The target network is a "frozen" copy that updates slowly, providing stable
learning targets. Without it, training is like trying to hit a moving target.

### 2. Experience Replay

Instead of learning from experiences as they happen (which would be correlated
and unstable), we store experiences in a replay buffer and sample random
batches. This breaks correlations and allows reusing good experiences.

### 3. Epsilon-Greedy Exploration

The agent faces an "exploration vs exploitation" dilemma:
- Exploitation: Always pick the action with highest Q-value (greedy)
- Exploration: Sometimes pick random actions to discover new strategies

Epsilon-greedy balances this:
- With probability ε (epsilon): take a RANDOM action (explore)
- With probability 1-ε: take the BEST action (exploit)

Epsilon starts high (lots of exploration) and decays over time as the
agent becomes more confident in its learned policy.

## Training Loop (conceptual):

    for episode in range(num_episodes):
        state = env.reset()  # Start new roast
        while not done:
            action = agent.select_action(state)  # ε-greedy
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()  # Learn from replay buffer
            state = next_state
        agent.decay_epsilon()  # Less exploration over time
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
    Deep Q-Network Agent for learning coffee roast control.

    This class ties together all DQN components into a cohesive agent
    that can learn from experience and make decisions.

    Key Components:
        policy_net: Neural network that learns Q-values (updated every step)
        target_net: Stable copy of policy_net (updated periodically)
        replay_buffer: Memory for storing and sampling experiences
        optimizer: Adam optimizer for updating policy_net weights

    Hyperparameters:
        gamma: Discount factor - how much to value future rewards
        epsilon: Exploration rate - probability of random action
        batch_size: Number of experiences to learn from per step
        target_update_freq: How often to update target network

    Actions (for coffee roaster - 9 combinations of heater and fan control):
        The agent controls TWO variables simultaneously:
        - Heater: decrease (-1), keep (0), or increase (+1)
        - Fan: decrease (-1), keep (0), or increase (+1)
        
        This creates a 3×3 = 9 action space:
            Action 0: heater -1, fan -1 (decrease both)
            Action 1: heater -1, fan  0 (decrease heater, keep fan)
            Action 2: heater -1, fan +1 (decrease heater, increase fan)
            Action 3: heater  0, fan -1 (keep heater, decrease fan)
            Action 4: heater  0, fan  0 (keep both - no change)
            Action 5: heater  0, fan +1 (keep heater, increase fan)
            Action 6: heater +1, fan -1 (increase heater, decrease fan)
            Action 7: heater +1, fan  0 (increase heater, keep fan)
            Action 8: heater +1, fan +1 (increase both)

    State Vector (5 elements):
        [bean_temp, ror, heater_level, fan_level, time]

    Example Usage:
        >>> agent = DQNAgent(state_size=5, action_size=9)
        >>> state = roaster.get_state_vector()  # [temp, ror, heater, fan, time]
        >>> action_idx = agent.select_action(state)  # 0-8
        >>> heater_change, fan_change = agent.get_action_value(action_idx)
        >>> next_state, reward, done = roaster.tick(heater_change, fan_change)
    """

    # Action mapping: neural network output index -> (heater_change, fan_change)
    # 9 actions = 3 heater options × 3 fan options
    # This is called a "discrete action space" with combined actions
    ACTION_MAP = {
        0: (-1, -1),  # Decrease heater, decrease fan
        1: (-1,  0),  # Decrease heater, keep fan
        2: (-1, +1),  # Decrease heater, increase fan
        3: ( 0, -1),  # Keep heater, decrease fan
        4: ( 0,  0),  # Keep both (no change)
        5: ( 0, +1),  # Keep heater, increase fan
        6: (+1, -1),  # Increase heater, decrease fan
        7: (+1,  0),  # Increase heater, keep fan
        8: (+1, +1),  # Increase both
    }

    def __init__(
        self,
        state_size: int = 5,
        action_size: int = 9,
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
        """
        Initialize the DQN agent with all hyperparameters.

        Args:
            state_size: Dimension of state vector (5: temp, ror, heater, fan, time)
            action_size: Number of possible actions (9: 3 heater × 3 fan options)
            hidden_size: Neurons in hidden layers (larger = more capacity)
            learning_rate: How fast to update weights (too high = unstable,
                          too low = slow learning). 1e-3 is a good default.
            gamma: Discount factor (0-1). Higher = care more about future rewards.
                   0.99 means future reward is worth 99% of immediate reward.
                   This makes the agent plan ahead rather than be greedy.
            epsilon_start: Initial exploration rate (1.0 = 100% random actions)
            epsilon_end: Minimum exploration rate (always keep some randomness)
            epsilon_decay: Multiply epsilon by this after each episode
                          0.995 means ~500 episodes to go from 1.0 to 0.01
            buffer_size: How many experiences to remember
            batch_size: How many experiences to learn from per training step
            target_update_freq: Update target network every N training steps
            device: 'cuda' for GPU, 'cpu' for CPU, or None for auto-detect
        """
        # Store configuration
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Discount factor for future rewards
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Exploration parameters (epsilon-greedy strategy)
        # Start with high exploration, gradually shift to exploitation
        self.epsilon = epsilon_start  # Current exploration rate
        self.epsilon_end = epsilon_end  # Minimum exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate per episode

        # Device setup: Use GPU if available for faster training
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ==== NEURAL NETWORKS ====
        # Policy network: The network we're actively training
        # This is what we use to select actions
        self.policy_net = QNetwork(state_size, action_size, hidden_size).to(self.device)

        # Target network: A stable copy for computing learning targets
        # Initialized with same weights as policy network
        self.target_net = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # Copy weights
        self.target_net.eval()  # Set to evaluation mode (no gradient tracking)

        # ==== TRAINING COMPONENTS ====
        # Adam optimizer: Efficient gradient descent with adaptive learning rate
        # Only optimizes policy_net - target_net is updated by copying weights
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay buffer: Memory for storing past experiences
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training step counter: Used to know when to update target network
        self.steps_done = 0

    def select_action(self, state: np.ndarray) -> int:
        """
        Select an action using epsilon-greedy policy.

        This is the core decision-making function. It balances:
        - EXPLORATION: Random actions to discover new strategies
        - EXPLOITATION: Best known action based on learned Q-values

        The balance is controlled by epsilon (ε):
        - With probability ε: Choose random action (explore)
        - With probability 1-ε: Choose action with highest Q-value (exploit)

        Args:
            state: Current state observation as numpy array
                   Shape: (state_size,) e.g., [150.0, 8.5, 6.0, 4.0, 180.0]
                   Elements: [bean_temp, ror, heater_level, fan_level, time]

        Returns:
            Action index: 0-8 representing one of 9 heater/fan combinations
            Use get_action_value() to convert to (heater_change, fan_change)

        Example:
            >>> state = np.array([150.0, 8.5, 6.0, 4.0, 180.0])
            >>> action = agent.select_action(state)  # Returns 0-8
            >>> heater_change, fan_change = agent.get_action_value(action)
        """
        # EXPLORATION: With probability epsilon, take a random action
        # This helps discover new strategies the agent hasn't tried
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # EXPLOITATION: Otherwise, use the learned Q-values to pick the best action
        # torch.no_grad() disables gradient computation for efficiency
        # (we only need gradients during training, not action selection)
        with torch.no_grad():
            # Convert numpy array to PyTorch tensor
            # unsqueeze(0) adds batch dimension: [4] -> [1, 4]
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # Get Q-values for all actions: shape [1, 3]
            q_values = self.policy_net(state_tensor)

            # Return the index of the highest Q-value (best action)
            # argmax finds the index, .item() converts tensor to Python int
            return q_values.argmax(dim=1).item()

    def get_action_value(self, action_idx: int) -> tuple[int, int]:
        """
        Convert neural network action index to actual control values.

        The network outputs indices 0-8, but we need actual control
        commands for both the heater and fan.

        Args:
            action_idx: Network output (0-8)

        Returns:
            Tuple of (heater_change, fan_change):
                heater_change: -1 (decrease), 0 (keep), or +1 (increase)
                fan_change: -1 (decrease), 0 (keep), or +1 (increase)

        Example:
            >>> agent.get_action_value(0)  # Returns (-1, -1) - decrease both
            >>> agent.get_action_value(4)  # Returns (0, 0) - keep both
            >>> agent.get_action_value(8)  # Returns (+1, +1) - increase both
        """
        return self.ACTION_MAP[action_idx]

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Store an experience in the replay buffer for later learning.

        Call this after every step in the environment. The agent will
        later sample from these stored experiences to learn.

        Args:
            state: State before the action [temp, ror, heater, time]
            action: Action taken (0, 1, or 2)
            reward: Reward received (positive = good, negative = bad)
            next_state: State after the action
            done: True if the roast ended (terminal state)
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> float:
        """
        Perform one training step: sample a batch and update the network.

        This is where the actual learning happens! The agent:
        1. Samples random experiences from the replay buffer
        2. Computes what Q-values it currently predicts
        3. Computes what Q-values it SHOULD predict (targets)
        4. Updates the network to reduce the difference

        The math behind this (Bellman equation):
            Q(s, a) = r + γ * max_a' Q(s', a')

        In words: The value of an action equals the immediate reward
        plus the discounted value of the best action in the next state.

        Returns:
            Loss value (lower = predictions closer to targets)
            Returns 0.0 if not enough experiences to train
        """
        # Don't train until we have enough experiences
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # ==== STEP 1: Sample random batch from replay buffer ====
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert numpy arrays to PyTorch tensors and move to GPU if available
        states = torch.FloatTensor(states).to(self.device)           # [batch, 4]
        actions = torch.LongTensor(actions).to(self.device)          # [batch]
        rewards = torch.FloatTensor(rewards).to(self.device)         # [batch]
        next_states = torch.FloatTensor(next_states).to(self.device) # [batch, 4]
        dones = torch.FloatTensor(dones).to(self.device)             # [batch]

        # ==== STEP 2: Compute current Q-values (what we predict) ====
        # policy_net(states) gives Q-values for ALL actions: [batch, 3]
        # .gather(1, actions) selects the Q-value for the action that was taken
        # Result: Q(s, a) for each experience in the batch
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # ==== STEP 3: Compute target Q-values (what we SHOULD predict) ====
        # This is the "ground truth" we're trying to match
        # Using Double DQN to prevent overestimation:
        with torch.no_grad():  # No gradients needed for target computation
            # DOUBLE DQN TRICK:
            # Standard DQN: uses target_net to both SELECT and EVALUATE best action
            # Double DQN: policy_net SELECTS, target_net EVALUATES
            # This reduces overestimation bias!

            # Step 3a: Use POLICY network to find the best action
            next_actions = self.policy_net(next_states).argmax(dim=1)

            # Step 3b: Use TARGET network to evaluate that action
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # Step 3c: Apply Bellman equation
            # target_q = reward + gamma * Q(next_state, best_action)
            # If done=True, there's no next state, so just use reward
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # ==== STEP 4: Compute loss and update network ====
        # Loss = how far our predictions are from targets
        # MSE = Mean Squared Error: average of (prediction - target)^2
        loss = nn.MSELoss()(current_q, target_q)

        # Zero out gradients from previous step (PyTorch accumulates gradients)
        self.optimizer.zero_grad()

        # Backpropagation: compute gradients of loss with respect to weights
        loss.backward()

        # Gradient clipping: prevent exploding gradients that destabilize training
        # If gradients are too large, scale them down to max_norm
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        # Apply gradients: update weights to reduce loss
        self.optimizer.step()

        # ==== STEP 5: Periodically update target network ====
        # Target network provides stable targets. If we updated it every step,
        # our targets would shift constantly (like aiming at a moving target).
        # Instead, we copy policy_net weights every N steps.
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()  # Return loss for logging/monitoring

    def update_target_network(self) -> None:
        """
        Copy weights from policy network to target network.

        The target network provides stable Q-value targets during training.
        We periodically sync it with the policy network (every target_update_freq steps).

        Why not update every step?
        - If targets change every step, training becomes unstable
        - It's like trying to hit a moving target
        - Periodic updates give the policy network time to converge

        Alternative: "soft" updates where we blend weights:
            target = tau * policy + (1 - tau) * target
        This is smoother but we use "hard" updates for simplicity.
        """
        # load_state_dict copies all weights and biases
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self) -> None:
        """
        Reduce exploration rate after each episode.

        As training progresses, the agent should explore less (it knows
        more about what works) and exploit more (use what it learned).

        Decay formula: epsilon = epsilon * decay_rate
        With decay=0.995, it takes ~500 episodes to go from 1.0 to 0.01

        The min() ensures we never go below epsilon_end, so we always
        maintain some exploration (important for adapting to changes).
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str) -> None:
        """
        Save the agent's complete state to a file.

        Saves everything needed to resume training or deploy the agent:
        - Network weights (the learned knowledge)
        - Optimizer state (momentum, adaptive learning rates)
        - Epsilon (current exploration rate)
        - Steps done (for target network update timing)

        Args:
            filepath: Where to save (e.g., 'models/dqn_roaster_v1.pth')

        Note:
            The replay buffer is NOT saved (too large, and not needed
            for deployment). For full training resumption, you'd need
            to save it separately.
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """
        Load a previously saved agent state.

        Restores all state from a checkpoint file, allowing you to:
        - Resume training from where you left off
        - Deploy a trained agent for inference
        - Fine-tune on new data

        Args:
            filepath: Path to saved checkpoint file

        Note:
            map_location ensures the model loads correctly even if
            it was saved on GPU but loaded on CPU (or vice versa).
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        print(f"Model loaded from {filepath}")
