"""
Proper Snake Training using Stable Baselines3
Shows the RIGHT way to train on Snake game with standard RL library
"""
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from core.planning_test_games import SnakeGame


class SnakeGymWrapper(gym.Env):
    """Wrap Snake game to work with Gymnasium/Stable-Baselines3"""

    def __init__(self, size=20, num_pellets=10):
        super().__init__()
        self.game = SnakeGame(size=size, num_pellets=num_pellets)

        # Define spaces
        # Observation: flattened game grid
        self.observation_space = gym.spaces.Box(
            low=0, high=3,
            shape=(size * size,),
            dtype=np.float32
        )

        # Actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = gym.spaces.Discrete(4)

        self.size = size
        self.current_state = None

    def _get_obs(self):
        """Convert game state to observation"""
        # Create grid: 0=empty, 1=food, 2=snake, 3=head
        grid = np.zeros((self.size, self.size), dtype=np.float32)

        # Add food
        for fx, fy in self.current_state.get('food_positions', []):
            if 0 <= fx < self.size and 0 <= fy < self.size:
                grid[fy, fx] = 1.0

        # Add snake body (if exists)
        body = self.current_state.get('snake_body', [])
        for bx, by in body:
            if 0 <= bx < self.size and 0 <= by < self.size:
                grid[by, bx] = 2.0

        # Add agent/head position
        agent_x, agent_y = self.current_state.get('agent_pos', (self.size//2, self.size//2))
        if 0 <= agent_x < self.size and 0 <= agent_y < self.size:
            grid[agent_y, agent_x] = 3.0

        return grid.flatten()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_state = self.game.reset()
        return self._get_obs(), {}

    def step(self, action):
        # Step the game
        self.current_state, reward, done = self.game.step(action)

        # Get observation
        obs = self._get_obs()

        # Reward shaping for better learning
        # Base reward from game (10 for pellet)
        shaped_reward = reward

        # Add small survival bonus
        if not done:
            shaped_reward += 0.01

        # Death penalty
        if done and reward <= 0:
            shaped_reward = -1.0

        info = {
            'score': self.current_state.get('score', 0),
            'pellets_collected': self.current_state.get('score', 0),
        }

        return obs, shaped_reward, done, False, info

    def render(self):
        """Simple ASCII render"""
        grid = [[' ' for _ in range(self.size)] for _ in range(self.size)]

        # Add food
        for fx, fy in self.current_state.get('food_positions', []):
            if 0 <= fx < self.size and 0 <= fy < self.size:
                grid[fy][fx] = '.'

        # Add snake body
        body = self.current_state.get('snake_body', [])
        for bx, by in body:
            if 0 <= bx < self.size and 0 <= by < self.size:
                grid[by][bx] = 'o'

        # Add head
        agent_x, agent_y = self.current_state.get('agent_pos', (self.size//2, self.size//2))
        if 0 <= agent_x < self.size and 0 <= agent_y < self.size:
            grid[agent_y][agent_x] = 'S'

        print('\n'.join(''.join(row) for row in grid))
        print()


class TrainingCallback(BaseCallback):
    """Callback to track training progress"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_scores = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self):
        # Track episode stats
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1

        # Check if episode done
        if self.locals['dones'][0]:
            score = self.locals['infos'][0].get('score', 0)
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_scores.append(score)
            self.episode_lengths.append(self.current_episode_length)

            # Print progress every episode
            if len(self.episode_scores) % 10 == 0:
                recent_scores = self.episode_scores[-100:] if len(self.episode_scores) >= 100 else self.episode_scores
                recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards

                print(f"\nEpisode {len(self.episode_scores)}")
                print(f"  Score: {score}")
                print(f"  Avg Score (last {len(recent_scores)}): {np.mean(recent_scores):.2f}")
                print(f"  Avg Reward (last {len(recent_rewards)}): {np.mean(recent_rewards):.2f}")
                print(f"  Best Score: {max(self.episode_scores)}")

            # Reset episode counters
            self.current_episode_reward = 0
            self.current_episode_length = 0

        return True


def train_snake(algorithm='PPO', total_timesteps=100000, save_path='snake_sb3_model'):
    """Train Snake with Stable-Baselines3"""

    print("=" * 70)
    print(f"TRAINING SNAKE WITH {algorithm}")
    print("=" * 70)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Expected episodes: ~{total_timesteps // 200} (assuming avg 200 steps/episode)")
    print()

    # Create environment
    env = SnakeGymWrapper(size=20, num_pellets=10)
    env = Monitor(env)  # Wrap with Monitor for better tracking

    # Create model
    print(f"Creating {algorithm} model...")
    if algorithm == 'PPO':
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=0
        )
    else:  # DQN
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=32,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1000,
            exploration_fraction=0.3,
            exploration_final_eps=0.01,
            verbose=0
        )

    print(f"Model created!")
    print(f"Policy architecture: {model.policy}")
    print()

    # Train
    print("Starting training...")
    print("-" * 70)
    callback = TrainingCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    # Final statistics
    print(f"\nTotal episodes: {len(callback.episode_scores)}")
    print(f"Average score: {np.mean(callback.episode_scores):.2f}")
    print(f"Best score: {max(callback.episode_scores)}")
    print(f"Final 100 avg: {np.mean(callback.episode_scores[-100:]):.2f}")

    # Save model
    model.save(save_path)
    print(f"\nModel saved to: {save_path}.zip")

    return model, env, callback


def test_model(model, env, num_episodes=5, render=True):
    """Test trained model"""

    print("\n" + "=" * 70)
    print("TESTING TRAINED MODEL")
    print("=" * 70)

    scores = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        print(f"\nEpisode {episode + 1}:")

        while not done and steps < 1000:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            if render and steps % 10 == 0:
                env.render()

        score = info.get('score', 0)
        scores.append(score)

        print(f"Final Score: {score}, Steps: {steps}, Reward: {episode_reward:.1f}")

        if render:
            print("\nFinal State:")
            env.render()

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"Average Score: {np.mean(scores):.2f} +/- {np.std(scores):.2f}")
    print(f"Best Score: {max(scores)}")
    print(f"Success Rate (>5 pellets): {sum(1 for s in scores if s >= 5) / len(scores) * 100:.1f}%")

    return scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='PPO', choices=['PPO', 'DQN'])
    parser.add_argument('--steps', type=int, default=100000, help='Total training timesteps')
    parser.add_argument('--test-only', type=str, help='Path to model to test (skip training)')
    parser.add_argument('--render', action='store_true', help='Render test episodes')
    args = parser.parse_args()

    if args.test_only:
        # Load and test existing model
        print(f"Loading model from {args.test_only}...")
        env = SnakeGymWrapper(size=20, num_pellets=10)

        if args.algo == 'PPO':
            model = PPO.load(args.test_only)
        else:
            model = DQN.load(args.test_only)

        test_model(model, env, num_episodes=10, render=args.render)
    else:
        # Train new model
        model, env, callback = train_snake(
            algorithm=args.algo,
            total_timesteps=args.steps,
            save_path=f'snake_{args.algo.lower()}_sb3'
        )

        # Test it
        print("\nNow testing the trained model...\n")
        test_model(model, env, num_episodes=5, render=True)
