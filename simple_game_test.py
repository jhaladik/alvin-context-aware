"""
Simple Game + Pre-trained Model Test
Creates a CartPole game and runs a pre-trained agent on it
"""
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from huggingface_sb3 import load_from_hub
import time

def test_pretrained_model():
    """Test pre-trained model on CartPole"""

    print("=" * 70)
    print("TESTING PRE-TRAINED MODEL ON SIMPLE GAME")
    print("=" * 70)
    print()

    # Create environment
    print("Creating CartPole environment...")
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    print(f"Environment: {env.spec.id}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()

    # Load pre-trained model
    print("Loading pre-trained PPO model from Hugging Face...")
    try:
        # Try to load pre-trained CartPole model
        checkpoint_path = load_from_hub(
            repo_id="sb3/ppo-CartPole-v1",
            filename="ppo-CartPole-v1.zip"
        )
        model = PPO.load(checkpoint_path)
        print("Successfully loaded pre-trained model!")
        print()

        # Print model details
        print("MODEL HYPERPARAMETERS:")
        print(f"  Learning rate: {model.learning_rate}")
        print(f"  Batch size: {model.batch_size}")
        print(f"  N steps: {model.n_steps}")
        print(f"  Gamma: {model.gamma}")
        print(f"  GAE lambda: {model.gae_lambda}")
        print(f"  Clip range: {model.clip_range}")
        print()

        # Print network architecture
        print("NETWORK ARCHITECTURE:")
        print(f"  Policy: {model.policy}")
        total_params = sum(p.numel() for p in model.policy.parameters())
        print(f"  Total parameters: {total_params:,}")
        print()

    except Exception as e:
        print(f"Could not load pre-trained model: {e}")
        print("Creating new untrained model for demonstration...")
        model = PPO("MlpPolicy", env, verbose=0)
        print()

    # Run episodes
    print("RUNNING TEST EPISODES")
    print("-" * 70)

    num_episodes = 5
    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0

        while not (done or truncated):
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, done, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            # Limit episode length for display
            if episode_length >= 500:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(f"Episode {episode + 1}: Reward = {episode_reward:.1f}, Length = {episode_length}")

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Average Reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.2f}")
    print()
    print("CartPole is 'solved' when average reward >= 475 over 100 episodes")
    print(f"This model achieved: {np.mean(episode_rewards):.1f}")

    if np.mean(episode_rewards) >= 475:
        print("SUCCESS! Model performs at solved level!")
    elif np.mean(episode_rewards) >= 200:
        print("GOOD! Model shows strong performance")
    else:
        print("Model needs more training or is untrained")

    env.close()


def test_custom_grid_game():
    """Test on a custom simple grid game"""

    print("\n" + "=" * 70)
    print("TESTING ON CUSTOM GRID GAME")
    print("=" * 70)
    print()

    # Simple 5x5 grid game: reach the goal
    class SimpleGridGame(gym.Env):
        """Simple grid navigation game"""

        def __init__(self, size=5):
            super().__init__()
            self.size = size
            self.observation_space = gym.spaces.Box(
                low=0, high=size-1, shape=(2,), dtype=np.float32
            )
            self.action_space = gym.spaces.Discrete(4)  # up, down, left, right
            self.reset()

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.agent_pos = np.array([0, 0], dtype=np.float32)
            self.goal_pos = np.array([self.size-1, self.size-1], dtype=np.float32)
            return self.agent_pos.copy(), {}

        def step(self, action):
            # Move agent
            if action == 0 and self.agent_pos[1] > 0:  # up
                self.agent_pos[1] -= 1
            elif action == 1 and self.agent_pos[1] < self.size - 1:  # down
                self.agent_pos[1] += 1
            elif action == 2 and self.agent_pos[0] > 0:  # left
                self.agent_pos[0] -= 1
            elif action == 3 and self.agent_pos[0] < self.size - 1:  # right
                self.agent_pos[0] += 1

            # Check if reached goal
            done = np.array_equal(self.agent_pos, self.goal_pos)
            reward = 10.0 if done else -0.1  # penalty for each step

            return self.agent_pos.copy(), reward, done, False, {}

        def render(self):
            grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
            grid[int(self.agent_pos[1])][int(self.agent_pos[0])] = 'A'
            grid[int(self.goal_pos[1])][int(self.goal_pos[0])] = 'G'
            print('\n'.join(' '.join(row) for row in grid))
            print()

    # Create custom environment
    print("Creating custom 5x5 grid game...")
    env = SimpleGridGame(size=5)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()

    # Train a quick agent (or use random policy)
    print("Training a quick PPO agent (10,000 steps)...")
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=10000)
    print("Training complete!")
    print()

    # Test the trained agent
    print("TESTING TRAINED AGENT")
    print("-" * 70)

    for episode in range(3):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        print(f"\nEpisode {episode + 1}:")
        env.render()

        while not done and episode_length < 20:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            time.sleep(0.2)  # Slow down for visualization
            env.render()

        print(f"Reward: {episode_reward:.1f}, Steps: {episode_length}")

        if done:
            print("GOAL REACHED!")
        else:
            print("Did not reach goal in 20 steps")


if __name__ == "__main__":
    # Test 1: Pre-trained CartPole
    test_pretrained_model()

    # Test 2: Custom grid game with quick training
    test_custom_grid_game()
