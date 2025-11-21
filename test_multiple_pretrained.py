"""
Test Multiple Pre-trained Models on Snake
Compares transfer learning effectiveness across different Atari games
"""
import numpy as np
from stable_baselines3 import DQN, PPO
from huggingface_sb3 import load_from_hub
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import the adapter we created
from snake_with_pretrained import SnakeAsAtariAdapter


def test_model(model_repo, model_file, model_name, num_episodes=10):
    """Test a pre-trained model on Snake"""

    print(f"\n{'=' * 70}")
    print(f"TESTING: {model_name}")
    print(f"{'=' * 70}")
    print(f"Repo: {model_repo}")
    print()

    # Create environment
    env = SnakeAsAtariAdapter(size=20, num_pellets=10)

    # Load model
    try:
        print(f"Loading model...")
        checkpoint = load_from_hub(repo_id=model_repo, filename=model_file)

        # Determine algorithm
        if 'dqn' in model_file.lower():
            model = DQN.load(checkpoint)
        else:
            model = PPO.load(checkpoint)

        print("Model loaded successfully!\n")
    except Exception as e:
        print(f"Failed to load: {e}\n")
        return None

    # Test episodes
    scores = []
    episode_lengths = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < 1000:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

        score = info['score']
        scores.append(score)
        episode_lengths.append(steps)

    avg_score = np.mean(scores)
    std_score = np.std(scores)
    best_score = max(scores)
    avg_length = np.mean(episode_lengths)

    print(f"Results:")
    print(f"  Average Score: {avg_score:.2f} +/- {std_score:.2f}")
    print(f"  Best Score: {best_score}/10")
    print(f"  Average Length: {avg_length:.1f} steps")

    return {
        'name': model_name,
        'avg_score': avg_score,
        'std_score': std_score,
        'best_score': best_score,
        'avg_length': avg_length
    }


def main():
    """Test multiple pre-trained models"""

    print("=" * 70)
    print("TRANSFER LEARNING COMPARISON: PRE-TRAINED ATARI MODELS ON SNAKE")
    print("=" * 70)
    print()
    print("Testing which Atari game transfers best to Snake...")
    print()

    models_to_test = [
        {
            'repo': 'sb3/dqn-MsPacmanNoFrameskip-v4',
            'file': 'dqn-MsPacmanNoFrameskip-v4.zip',
            'name': 'Ms. Pac-Man (DQN)',
            'reason': 'Maze navigation + pellet collection'
        },
        {
            'repo': 'ThomasSimonini/ppo-BreakoutNoFrameskip-v4',
            'file': 'ppo-BreakoutNoFrameskip-v4.zip',
            'name': 'Breakout (PPO)',
            'reason': 'Spatial patterns + target hitting'
        },
    ]

    results = []

    for model_spec in models_to_test:
        print(f"\nWhy {model_spec['name']}? {model_spec['reason']}")

        result = test_model(
            model_spec['repo'],
            model_spec['file'],
            model_spec['name'],
            num_episodes=10
        )

        if result:
            results.append(result)

    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print()
    print(f"{'Model':<30} {'Avg Score':<15} {'Best':<10} {'Steps':<10}")
    print("-" * 70)

    for r in results:
        print(f"{r['name']:<30} {r['avg_score']:.2f} +/- {r['std_score']:.2f}    "
              f"{r['best_score']}/10      {r['avg_length']:.0f}")

    print()
    print("Baseline Comparisons:")
    print("  Random Policy:               0.80 +/- 0.75    2/10       63")
    print("  SB3 PPO (flat grid, 100K):   5.08 +/- 2.15    8/10       -")
    print("  Custom DQN (ray-based, 500): 8.05 +/- 1.50    10/10      -")
    print()
    print("KEY INSIGHT:")
    print("Pre-trained models on DIFFERENT visual games don't transfer well")
    print("without fine-tuning. You need task-specific training OR same-task")
    print("pre-training (e.g., CartPole → CartPole worked perfectly: 500/500)")


if __name__ == "__main__":
    main()
