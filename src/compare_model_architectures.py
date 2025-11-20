"""
Comprehensive Model Architecture Comparison

Compares two trained model variants:
1. BASELINE: 8 rays × 10 tiles (92 dims) - smaller world view
2. EXPANDED: 16 rays × 15 tiles (180 dims) - bigger world, longer view

Tests both models on same games and generates detailed comparison report.

Usage:
    python compare_model_architectures.py \\
        --baseline checkpoints/baseline_model_policy.pth \\
        --expanded checkpoints/faith_evolution_20251120_091144_final_policy.pth \\
        --episodes 50
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))

import torch
import numpy as np
import argparse
from collections import defaultdict

# Import both observer types
from core.temporal_observer import TemporalFlowObserver  # Baseline: 92 dims
from core.expanded_temporal_observer import ExpandedTemporalObserver  # Expanded: 180 dims

# Games
from core.planning_test_games import SnakeGame, PacManGame, DungeonGame

# Agent
from context_aware_agent import (
    ContextAwareDQN,
    infer_context_from_observation,
    add_context_to_observation
)

# Revolutionary systems
from core.faith_system import FaithPopulation
from core.entity_discovery import EntityDiscoveryWorldModel
from core.world_model import WorldModelNetwork


def detect_obs_dim(checkpoint):
    """Detect observation dimension from checkpoint"""
    state_dict = checkpoint['policy_net']

    # Try different possible first layer keys
    possible_keys = [
        'perception_net.0.weight',
        'q_heads.snake.0.weight',
        'fc1.weight',
    ]

    for key in possible_keys:
        if key in state_dict:
            return state_dict[key].shape[1]

    raise ValueError(f"Could not detect obs_dim from checkpoint. Keys: {list(state_dict.keys())[:10]}")


def load_model(policy_path, obs_dim=None):
    """Load model with specified or auto-detected observation dimension"""
    print(f"\nLoading model from: {policy_path}")

    checkpoint = torch.load(policy_path, map_location='cpu', weights_only=False)

    # Auto-detect obs_dim if not provided
    if obs_dim is None:
        obs_dim = detect_obs_dim(checkpoint)
        print(f"  Auto-detected obs_dim: {obs_dim}")

    # Create and load policy network
    agent = ContextAwareDQN(obs_dim=obs_dim, action_dim=4)
    agent.load_state_dict(checkpoint['policy_net'])
    agent.eval()

    # Training info
    episodes_trained = len(checkpoint.get('episode_rewards', []))
    steps_done = checkpoint.get('steps_done', 0)

    print(f"  Architecture: {obs_dim} dims")
    print(f"  Episodes trained: {episodes_trained}")
    print(f"  Steps: {steps_done:,}")

    # Load world model if available
    world_model = None
    base_path = policy_path.replace('_policy.pth', '').replace('_best.pth', '').replace('_final.pth', '')
    world_model_path = f"{base_path}_world_model.pth"

    if os.path.exists(world_model_path):
        wm_checkpoint = torch.load(world_model_path, map_location='cpu', weights_only=False)
        state_dict = wm_checkpoint['model']

        # Detect hidden_dim
        if 'state_predictor.0.weight' in state_dict:
            hidden_dim = state_dict['state_predictor.0.weight'].shape[0]
        else:
            hidden_dim = 256

        world_model = WorldModelNetwork(state_dim=obs_dim, action_dim=4, hidden_dim=hidden_dim)
        world_model.load_state_dict(state_dict)
        world_model.eval()
        print(f"  World model loaded: hidden_dim={hidden_dim}")
    else:
        print(f"  World model not found")

    return agent, world_model, checkpoint


def test_model(agent, world_model, observer, game, game_name, num_episodes=50,
               planning_freq=0.2, planning_horizon=20):
    """Test model on game"""
    scores = []
    steps_list = []
    context_counts = {'snake': 0, 'balanced': 0, 'survival': 0}

    for episode in range(num_episodes):
        game.reset()
        observer.reset()
        game_state = game._get_game_state()

        total_reward = 0
        steps = 0
        done = False

        while not done and steps < 1000:
            # Get observation
            obs = observer.observe(game_state)

            # Infer context
            context_vector = infer_context_from_observation(obs)

            # Track context
            if context_vector[0] == 1.0:
                context_counts['snake'] += 1
            elif context_vector[1] == 1.0:
                context_counts['balanced'] += 1
            else:
                context_counts['survival'] += 1

            # Add context
            obs_with_context = add_context_to_observation(obs, context_vector)

            # Select action (greedy for testing)
            action = agent.get_action(obs_with_context, epsilon=0.0)

            # Execute
            game_state, reward, done = game.step(action)
            total_reward += reward
            steps += 1

        scores.append(game_state['score'])
        steps_list.append(steps)

    return {
        'scores': scores,
        'steps': steps_list,
        'avg_score': np.mean(scores),
        'std_score': np.std(scores),
        'max_score': max(scores),
        'min_score': min(scores),
        'avg_steps': np.mean(steps_list),
        'context_counts': context_counts
    }


def print_comparison_table(baseline_results, expanded_results):
    """Print side-by-side comparison"""
    print(f"\n{'='*100}")
    print(f"DETAILED COMPARISON: BASELINE (8×10) vs EXPANDED (16×15)")
    print(f"{'='*100}")

    games = ['snake', 'pacman', 'dungeon']

    # Header
    print(f"\n{'Game':<15} {'Metric':<20} {'Baseline (92d)':<25} {'Expanded (180d)':<25} {'Δ Change':<15}")
    print(f"{'-'*100}")

    for game_name in games:
        if game_name not in baseline_results or game_name not in expanded_results:
            continue

        base = baseline_results[game_name]
        exp = expanded_results[game_name]

        # Average score
        delta_score = exp['avg_score'] - base['avg_score']
        pct_change = (delta_score / base['avg_score'] * 100) if base['avg_score'] != 0 else 0
        print(f"{game_name.capitalize():<15} {'Avg Score':<20} {base['avg_score']:>8.2f} ± {base['std_score']:<6.2f}    "
              f"{exp['avg_score']:>8.2f} ± {exp['std_score']:<6.2f}    "
              f"{delta_score:>+6.2f} ({pct_change:>+5.1f}%)")

        # Max score
        delta_max = exp['max_score'] - base['max_score']
        print(f"{'':15} {'Max Score':<20} {base['max_score']:>8}                "
              f"{exp['max_score']:>8}                "
              f"{delta_max:>+6}")

        # Average steps
        delta_steps = exp['avg_steps'] - base['avg_steps']
        pct_steps = (delta_steps / base['avg_steps'] * 100) if base['avg_steps'] != 0 else 0
        print(f"{'':15} {'Avg Steps':<20} {base['avg_steps']:>8.1f}               "
              f"{exp['avg_steps']:>8.1f}               "
              f"{delta_steps:>+6.1f} ({pct_steps:>+5.1f}%)")

        print()


def main():
    parser = argparse.ArgumentParser(description='Compare Baseline vs Expanded Model Architectures')
    parser.add_argument('--baseline', type=str, required=True, help='Baseline model path (92 dims)')
    parser.add_argument('--expanded', type=str, required=True, help='Expanded model path (180 dims)')
    parser.add_argument('--episodes', type=int, default=50, help='Episodes per game')
    parser.add_argument('--game', choices=['snake', 'pacman', 'dungeon', 'all'], default='all')

    args = parser.parse_args()

    print(f"{'='*100}")
    print(f"MODEL ARCHITECTURE COMPARISON")
    print(f"{'='*100}")
    print(f"\nConfiguration:")
    print(f"  Episodes per game: {args.episodes}")
    print(f"  Games to test: {args.game}")

    # Load models
    print(f"\n{'='*100}")
    print(f"LOADING MODELS")
    print(f"{'='*100}")

    print(f"\n[1] BASELINE MODEL (Smaller World View)")
    baseline_agent, baseline_wm, baseline_checkpoint = load_model(args.baseline)  # Auto-detect

    print(f"\n[2] EXPANDED MODEL (Bigger World, Longer View)")
    expanded_agent, expanded_wm, expanded_checkpoint = load_model(args.expanded)  # Auto-detect

    # Create observers based on detected dimensions
    baseline_obs_dim = detect_obs_dim(baseline_checkpoint)
    expanded_obs_dim = detect_obs_dim(expanded_checkpoint)

    # Create appropriate observers
    if baseline_obs_dim <= 100:
        baseline_observer = TemporalFlowObserver(num_rays=8, ray_length=10)
        baseline_type = "Baseline (8×10)"
    else:
        baseline_observer = ExpandedTemporalObserver(num_rays=16, ray_length=15)
        baseline_type = "Expanded (16×15)"

    if expanded_obs_dim <= 100:
        expanded_observer = TemporalFlowObserver(num_rays=8, ray_length=10)
        expanded_type = "Baseline (8×10)"
    else:
        expanded_observer = ExpandedTemporalObserver(num_rays=16, ray_length=15)
        expanded_type = "Expanded (16×15)"

    print(f"\n{'='*100}")
    print(f"OBSERVER COMPARISON")
    print(f"{'='*100}")
    print(f"\nBaseline Model Observer: {baseline_type}")
    print(f"  Observation dims: {baseline_observer.obs_dim}")

    print(f"\nExpanded Model Observer: {expanded_type}")
    print(f"  Observation dims: {expanded_observer.obs_dim}")

    # Test games
    games_to_test = ['snake', 'pacman', 'dungeon'] if args.game == 'all' else [args.game]

    baseline_results = {}
    expanded_results = {}

    for game_name in games_to_test:
        print(f"\n{'='*100}")
        print(f"TESTING: {game_name.upper()}")
        print(f"{'='*100}")

        # Create game
        if game_name == 'snake':
            game = SnakeGame(size=20)
        elif game_name == 'pacman':
            game = PacManGame(size=20)
        else:
            game = DungeonGame(size=20)

        # Test baseline
        print(f"\n[1] Testing BASELINE model...")
        baseline_results[game_name] = test_model(
            baseline_agent, baseline_wm, baseline_observer,
            game, game_name, args.episodes
        )
        print(f"  Avg score: {baseline_results[game_name]['avg_score']:.2f}")

        # Test expanded
        print(f"\n[2] Testing EXPANDED model...")
        expanded_results[game_name] = test_model(
            expanded_agent, expanded_wm, expanded_observer,
            game, game_name, args.episodes
        )
        print(f"  Avg score: {expanded_results[game_name]['avg_score']:.2f}")

    # Print comparison
    print_comparison_table(baseline_results, expanded_results)

    # Summary verdict
    print(f"\n{'='*100}")
    print(f"VERDICT")
    print(f"{'='*100}")

    total_baseline_score = sum(r['avg_score'] for r in baseline_results.values())
    total_expanded_score = sum(r['avg_score'] for r in expanded_results.values())
    improvement = ((total_expanded_score - total_baseline_score) / total_baseline_score * 100)

    print(f"\nOverall Performance:")
    print(f"  Baseline total:  {total_baseline_score:.2f}")
    print(f"  Expanded total:  {total_expanded_score:.2f}")
    print(f"  Improvement:     {improvement:>+.1f}%")

    if improvement > 10:
        print(f"\n  ✅ EXPANDED model shows SIGNIFICANT improvement!")
        print(f"     Recommendation: Use EXPANDED architecture for warehouse application")
    elif improvement > 0:
        print(f"\n  ⚠️  EXPANDED model shows modest improvement")
        print(f"     Recommendation: Continue training or use expanded for complex scenarios")
    else:
        print(f"\n  ❌ BASELINE model performs better (or equal)")
        print(f"     Recommendation: More training needed for expanded model")

    # Save detailed report
    report_path = "model_comparison_report.txt"
    with open(report_path, 'w') as f:
        f.write(f"MODEL ARCHITECTURE COMPARISON REPORT\n")
        f.write(f"{'='*100}\n\n")
        f.write(f"Date: {os.popen('date').read()}\n")
        f.write(f"Episodes per game: {args.episodes}\n\n")

        f.write(f"BASELINE MODEL: {args.baseline}\n")
        f.write(f"  Architecture: 8 rays × 10 tiles (92 dims + 3 context = 95 total)\n")
        f.write(f"  Episodes trained: {len(baseline_checkpoint.get('episode_rewards', []))}\n\n")

        f.write(f"EXPANDED MODEL: {args.expanded}\n")
        f.write(f"  Architecture: 16 rays × 15 tiles (180 dims + 3 context = 183 total)\n")
        f.write(f"  Episodes trained: {len(expanded_checkpoint.get('episode_rewards', []))}\n\n")

        f.write(f"RESULTS BY GAME:\n")
        f.write(f"{'-'*100}\n")
        for game_name in games_to_test:
            if game_name in baseline_results:
                base = baseline_results[game_name]
                exp = expanded_results[game_name]
                f.write(f"\n{game_name.upper()}:\n")
                f.write(f"  Baseline: {base['avg_score']:.2f} ± {base['std_score']:.2f} (max: {base['max_score']})\n")
                f.write(f"  Expanded: {exp['avg_score']:.2f} ± {exp['std_score']:.2f} (max: {exp['max_score']})\n")
                delta = exp['avg_score'] - base['avg_score']
                pct = (delta / base['avg_score'] * 100) if base['avg_score'] != 0 else 0
                f.write(f"  Change:   {delta:>+.2f} ({pct:>+.1f}%)\n")

        f.write(f"\n\nVERDICT: Expanded model improvement = {improvement:>+.1f}%\n")

    print(f"\nDetailed report saved to: {report_path}")


if __name__ == '__main__':
    main()
