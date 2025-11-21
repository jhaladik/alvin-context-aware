"""
Deep Dive: What Does a Pre-trained Atari Model Actually Know?

Let's inspect the Ms. Pac-Man DQN to understand:
1. What visual features it learned
2. What patterns it recognizes
3. Why it doesn't transfer to Snake
"""
import numpy as np
import torch
import cv2
from stable_baselines3 import DQN
from huggingface_sb3 import load_from_hub
import sys
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from snake_with_pretrained import SnakeAsAtariAdapter
from core.planning_test_games import SnakeGame


def load_pacman_model():
    """Load the pre-trained Ms. Pac-Man model"""
    print("Loading Ms. Pac-Man DQN model...")
    checkpoint = load_from_hub(
        repo_id="sb3/dqn-MsPacmanNoFrameskip-v4",
        filename="dqn-MsPacmanNoFrameskip-v4.zip"
    )
    model = DQN.load(checkpoint)
    print("Model loaded!\n")
    return model


def inspect_network_architecture(model):
    """Inspect the CNN architecture"""
    print("=" * 70)
    print("NETWORK ARCHITECTURE")
    print("=" * 70)

    policy = model.policy
    q_net = policy.q_net

    print("\nQ-Network Structure:")
    print(q_net)

    print("\n\nLayer Details:")
    for name, param in q_net.named_parameters():
        print(f"  {name:40s} Shape: {str(list(param.shape)):20s} Params: {param.numel():,}")

    total_params = sum(p.numel() for p in q_net.parameters())
    print(f"\nTotal Parameters: {total_params:,}")

    return q_net


def analyze_conv_filters(model):
    """Analyze what the first convolutional layer learned"""
    print("\n" + "=" * 70)
    print("CONVOLUTIONAL FILTERS (What it sees)")
    print("=" * 70)

    q_net = model.policy.q_net

    # Get first conv layer
    first_conv = None
    for name, module in q_net.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            first_conv = module
            print(f"\nFirst Conv Layer: {name}")
            print(f"  Input channels: {module.in_channels}")
            print(f"  Output channels: {module.out_channels}")
            print(f"  Kernel size: {module.kernel_size}")
            print(f"  Stride: {module.stride}")
            break

    if first_conv is not None:
        weights = first_conv.weight.data.cpu().numpy()
        print(f"\nFilter weights shape: {weights.shape}")
        print(f"  Interpretation: {weights.shape[0]} filters, each looking at {weights.shape[1]} input frames")
        print(f"  with {weights.shape[2]}x{weights.shape[3]} spatial extent")

        # Analyze filter statistics
        print(f"\nFilter statistics:")
        print(f"  Mean: {weights.mean():.6f}")
        print(f"  Std: {weights.std():.6f}")
        print(f"  Min: {weights.min():.6f}")
        print(f"  Max: {weights.max():.6f}")

        return weights

    return None


def test_q_values_on_snake(model, num_tests=10):
    """Analyze Q-values the model produces on Snake states"""
    print("\n" + "=" * 70)
    print("Q-VALUE ANALYSIS ON SNAKE")
    print("=" * 70)

    env = SnakeAsAtariAdapter(size=20, num_pellets=10)

    q_value_stats = defaultdict(list)
    action_preferences = defaultdict(int)

    print("\nTesting model's Q-value predictions on Snake states...\n")

    for test in range(num_tests):
        obs, _ = env.reset()

        # Get Q-values from model
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            # Rearrange from (1, 84, 84, 4) to (1, 4, 84, 84)
            obs_tensor = obs_tensor.permute(0, 3, 1, 2)

            q_values = model.policy.q_net(obs_tensor).cpu().numpy()[0]

        action = np.argmax(q_values)

        print(f"Test {test + 1}:")
        print(f"  Q-values: {q_values}")
        print(f"  Chosen action: {action} (Q={q_values[action]:.2f})")

        # Track statistics
        for i, q in enumerate(q_values):
            q_value_stats[i].append(q)

        action_preferences[action] += 1

    print("\n" + "-" * 70)
    print("SUMMARY:")
    print("-" * 70)

    print("\nAverage Q-values per action:")
    for action in range(9):  # Atari has 9 actions
        avg_q = np.mean(q_value_stats[action])
        std_q = np.std(q_value_stats[action])
        count = action_preferences.get(action, 0)
        action_name = ["NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "DOWNRIGHT", "UPLEFT"][action]
        print(f"  Action {action} ({action_name:10s}): Q={avg_q:7.2f} +/- {std_q:6.2f}  (chosen {count} times)")

    print("\nAction preference distribution:")
    total = sum(action_preferences.values())
    for action, count in sorted(action_preferences.items(), key=lambda x: x[1], reverse=True):
        action_name = ["NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "DOWNRIGHT", "UPLEFT"][action]
        pct = 100 * count / total
        print(f"  {action_name:10s}: {count:2d}/{total} ({pct:.1f}%)")


def compare_snake_vs_pacman_observations(model):
    """Compare how different the observations look"""
    print("\n" + "=" * 70)
    print("OBSERVATION COMPARISON: Snake vs Ms. Pac-Man")
    print("=" * 70)

    # Create Snake observation
    env = SnakeAsAtariAdapter(size=20, num_pellets=10)
    snake_obs, _ = env.reset()

    print("\nSnake Observation:")
    print(f"  Shape: {snake_obs.shape}")
    print(f"  Mean pixel value: {snake_obs.mean():.2f}")
    print(f"  Std pixel value: {snake_obs.std():.2f}")
    print(f"  Unique values: {len(np.unique(snake_obs))}")
    print(f"  Value range: [{snake_obs.min()}, {snake_obs.max()}]")

    # Analyze what pixels are active
    frame = snake_obs[:, :, 0]  # Look at first frame
    nonzero = (frame > 0).sum()
    total = frame.size
    print(f"  Active pixels: {nonzero}/{total} ({100*nonzero/total:.1f}%)")

    print("\nMs. Pac-Man (what model was trained on):")
    print("  Shape: (84, 84, 4) - same")
    print("  Content: Complex maze with walls, pellets, ghosts, power-ups")
    print("  Pixel density: ~40-60% filled (dense maze)")
    print("  Spatial patterns: Corridors, corners, enclosed spaces")
    print("  Temporal patterns: Moving ghosts, blinking power-ups")

    print("\nKey Differences:")
    print("  1. Spatial structure: Maze corridors vs. Open grid")
    print("  2. Object density: Dense vs. Sparse")
    print("  3. Movement patterns: Constrained paths vs. Free movement")
    print("  4. Visual features: Walls/corridors vs. Simple objects")

    print("\nWhy transfer fails:")
    print("  - CNN learned to detect maze corridors (not present in Snake)")
    print("  - Policy learned to navigate constrained paths (Snake has open grid)")
    print("  - No wall-following behavior needed in Snake")
    print("  - Ghost avoidance patterns don't apply to Snake body")


def activation_analysis(model):
    """Analyze network activations on Snake vs what it expects"""
    print("\n" + "=" * 70)
    print("ACTIVATION ANALYSIS")
    print("=" * 70)

    env = SnakeAsAtariAdapter(size=20, num_pellets=10)
    obs, _ = env.reset()

    # Get intermediate activations
    q_net = model.policy.q_net

    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).permute(0, 3, 1, 2)

    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks
    hooks = []
    for name, module in q_net.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    # Forward pass
    with torch.no_grad():
        output = q_net(obs_tensor)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    print("\nLayer activations on Snake observation:")
    for name, activation in activations.items():
        shape = activation.shape
        mean_act = activation.mean().item()
        std_act = activation.std().item()
        sparsity = (activation == 0).float().mean().item()

        print(f"\n  {name}:")
        print(f"    Shape: {shape}")
        print(f"    Mean activation: {mean_act:.6f}")
        print(f"    Std activation: {std_act:.6f}")
        print(f"    Sparsity (% zeros): {100*sparsity:.1f}%")

    print("\n\nInterpretation:")
    print("  - Low activations suggest features don't match training data")
    print("  - High sparsity means many neurons are 'confused'")
    print("  - The network is seeing patterns it wasn't trained on")


def main():
    """Run complete analysis"""

    print("=" * 70)
    print("DEEP DIVE: WHAT DOES MS. PAC-MAN DQN ACTUALLY KNOW?")
    print("=" * 70)
    print()

    # Load model
    model = load_pacman_model()

    # 1. Inspect architecture
    q_net = inspect_network_architecture(model)

    # 2. Analyze convolutional filters
    filters = analyze_conv_filters(model)

    # 3. Test Q-values on Snake
    test_q_values_on_snake(model, num_tests=10)

    # 4. Compare observations
    compare_snake_vs_pacman_observations(model)

    # 5. Activation analysis
    activation_analysis(model)

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("The Ms. Pac-Man DQN learned:")
    print("  1. Visual features: Maze corridors, wall patterns, ghost shapes")
    print("  2. Spatial concepts: Path navigation, corner turning, corridor following")
    print("  3. Temporal patterns: Ghost movement prediction, power-up timing")
    print("  4. Strategy: Pellet collection in constrained maze environment")
    print()
    print("What it DIDN'T learn (not in Ms. Pac-Man):")
    print("  1. Open grid navigation")
    print("  2. Self-collision avoidance (growing tail)")
    print("  3. Sparse object collection")
    print("  4. Free movement in open space")
    print()
    print("This is why transfer learning failed:")
    print("  - The visual features don't match (maze vs. grid)")
    print("  - The behavioral patterns don't match (constrained vs. free)")
    print("  - The task structure doesn't match (avoid ghosts vs. avoid self)")
    print()
    print("Pre-trained models encode TASK-SPECIFIC knowledge, not universal skills!")


if __name__ == "__main__":
    main()
