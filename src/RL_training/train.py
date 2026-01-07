"""
Training loop for the DQN agent on the coffee roaster simulator.
"""

import logging
from typing import List

import numpy as np

from RL_training.DQN.dqn_agent import DQNAgent
from RL_training.environment import create_roaster


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
    logging.getLogger('RL_training.aillio_dummy').setLevel(logging.WARNING)

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
