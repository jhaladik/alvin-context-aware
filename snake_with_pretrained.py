"""
Snake Game Adapter for Pre-trained Ms. Pac-Man DQN Model
Adapts your Snake game to work with sb3/dqn-MsPacmanNoFrameskip-v4
NO TRAINING REQUIRED - uses pre-trained model directly!
"""
import gymnasium as gym
import numpy as np
import cv2
import sys
import os
from stable_baselines3 import DQN
from huggingface_sb3 import load_from_hub
from collections import deque

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from core.planning_test_games import SnakeGame


class SnakeAsAtariAdapter(gym.Env):
    """
    Adapter: Makes Snake look like Atari Ms. Pac-Man

    What Ms. Pac-Man DQN expects:
    - Observation: (84, 84, 4) grayscale images, 4-frame stack
    - Action space: Discrete(9) but we only use 4 directions
    - Preprocessing: Atari standard (grayscale, resize, stack)
    """

    def __init__(self, size=20, num_pellets=10):
        super().__init__()

        # Match Atari preprocessing
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(84, 84, 4),  # ← What pre-trained model expects
            dtype=np.uint8
        )

        # Atari Ms. Pac-Man has 9 actions, but we only use 4
        self.action_space = gym.spaces.Discrete(9)

        # Snake game
        self.game = SnakeGame(size=size, num_pellets=num_pellets)
        self.size = size

        # Frame buffer for stacking
        self.frame_buffer = deque(maxlen=4)

        # Action mapping: Atari -> Snake
        # Atari actions: [NOOP=0, FIRE=1, UP=2, RIGHT=3, LEFT=4, DOWN=5, ...]
        # Snake actions: [UP=0, DOWN=1, LEFT=2, RIGHT=3]
        self.action_map = {
            0: 0,  # NOOP -> UP (default)
            1: 0,  # FIRE -> UP
            2: 0,  # UP -> UP
            3: 3,  # RIGHT -> RIGHT
            4: 2,  # LEFT -> LEFT
            5: 1,  # DOWN -> DOWN
            6: 0,  # UPRIGHT -> UP
            7: 1,  # DOWNRIGHT -> DOWN
            8: 0,  # UPLEFT -> UP
        }

        print("Snake -> Atari Adapter Initialized!")
        print(f"  Observation space: {self.observation_space.shape}")
        print(f"  Action space: {self.action_space}")
        print(f"  Snake size: {size}x{size}")
        print(f"  Pellets: {num_pellets}")

    def _render_as_atari_frame(self, state):
        """
        Convert Snake state to 84x84 grayscale image (like Atari)

        Colors in grayscale:
        - Background: 0 (black)
        - Food: 255 (white - like Pac-Man pellets)
        - Snake body: 128 (gray)
        - Snake head: 200 (bright gray - like Pac-Man)
        """
        # Create 20x20 grid first
        grid = np.zeros((self.size, self.size), dtype=np.uint8)

        # Draw food (bright white like pellets)
        for fx, fy in state.get('food_positions', []):
            if 0 <= fx < self.size and 0 <= fy < self.size:
                grid[fy, fx] = 255

        # Draw snake body (gray)
        for bx, by in state.get('snake_body', []):
            if 0 <= bx < self.size and 0 <= by < self.size:
                grid[by, bx] = 128

        # Draw snake head (bright - like Pac-Man)
        ax, ay = state['agent_pos']
        if 0 <= ax < self.size and 0 <= ay < self.size:
            grid[ay, ax] = 200

        # Resize to 84x84 using OpenCV (same as Atari preprocessing)
        frame = cv2.resize(grid, (84, 84), interpolation=cv2.INTER_NEAREST)

        return frame

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset Snake game
        state = self.game.reset()

        # Get first frame
        frame = self._render_as_atari_frame(state)

        # Initialize frame buffer with 4 copies of first frame
        self.frame_buffer.clear()
        for _ in range(4):
            self.frame_buffer.append(frame)

        # Stack frames: (84, 84, 4)
        obs = np.stack(self.frame_buffer, axis=-1)

        return obs, {}

    def step(self, action):
        # Map Atari action to Snake action
        # Convert numpy array to int if needed
        action_int = int(action) if hasattr(action, '__iter__') else action
        snake_action = self.action_map.get(action_int, 0)

        # Step Snake game
        state, reward, done = self.game.step(snake_action)

        # Render new frame
        frame = self._render_as_atari_frame(state)

        # Add to buffer (automatically pops oldest)
        self.frame_buffer.append(frame)

        # Stack frames
        obs = np.stack(self.frame_buffer, axis=-1)

        # Shape reward to match Atari scale (Pac-Man gets 10 per pellet)
        # Snake base reward is 10, so we're already similar!
        shaped_reward = float(reward)

        info = {
            'score': state.get('score', 0),
            'snake_action': snake_action,
            'atari_action': action
        }

        return obs, shaped_reward, done, False, info


def test_pretrained_on_snake(num_episodes=10):
    """Load pre-trained Ms. Pac-Man DQN and test on Snake"""

    print("=" * 70)
    print("SNAKE GAME WITH PRE-TRAINED MS. PAC-MAN DQN")
    print("=" * 70)
    print()

    # Create adapted Snake environment
    env = SnakeAsAtariAdapter(size=20, num_pellets=10)
    print()

    # Load pre-trained Ms. Pac-Man model
    print("Loading pre-trained Ms. Pac-Man DQN model...")
    print("  Model: sb3/dqn-MsPacmanNoFrameskip-v4")
    print("  Training: 10M timesteps on Atari Ms. Pac-Man")
    print("  Performance: 2682 +/- 475 reward")
    print()

    try:
        checkpoint = load_from_hub(
            repo_id="sb3/dqn-MsPacmanNoFrameskip-v4",
            filename="dqn-MsPacmanNoFrameskip-v4.zip"
        )
        model = DQN.load(checkpoint)
        print("Model loaded successfully!")
        print()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nFalling back to random policy...")
        model = None

    # Test on Snake
    print("TESTING ON SNAKE")
    print("-" * 70)

    scores = []
    episode_lengths = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < 1000:
            if model is not None:
                action, _states = model.predict(obs, deterministic=True)
            else:
                # Random policy
                action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

        score = info['score']
        scores.append(score)
        episode_lengths.append(steps)

        print(f"Episode {episode + 1}: Score = {score}, "
              f"Steps = {steps}, Reward = {episode_reward:.1f}")

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"Average Score: {np.mean(scores):.2f} +/- {np.std(scores):.2f}")
    print(f"Best Score: {max(scores)}")
    print(f"Average Length: {np.mean(episode_lengths):.1f}")
    print()
    print("Comparison:")
    print(f"  Your custom DQN (500 episodes):  8.05/10 pellets")
    print(f"  SB3 PPO (100K steps, flat obs):  5.08/10 pellets")
    print(f"  Pre-trained Ms. Pac-Man (NOW):   {np.mean(scores):.2f}/10 pellets")
    print()
    print("Expected: The Ms. Pac-Man model should show some navigation skills")
    print("since it learned to collect pellets and avoid enemies in a similar")
    print("grid-based environment!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of test episodes')
    args = parser.parse_args()

    test_pretrained_on_snake(num_episodes=args.episodes)
