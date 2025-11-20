"""
Direct Comparison: Old Model vs Fixed Model

Compares the old model (faith_evolution_20251120_091144) with the fixed model
(faith_fixed_20251120_162417) on same tasks with same configuration.

Shows the impact of the world model bottleneck fix.

Usage:
    python compare_old_vs_fixed.py --episodes 30
    python compare_old_vs_fixed.py --quick  # 10 episodes for quick results
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))

import torch
import numpy as np
import argparse

from context_aware_agent import ContextAwareDQN, infer_context_from_observation, add_context_to_observation
from core.planning_test_games import SnakeGame, PacManGame, DungeonGame
from core.expanded_temporal_observer import ExpandedTemporalObserver
from core.world_model import WorldModelNetwork
from core.context_aware_world_model import ContextAwareWorldModel


def load_model_and_world_model(model_path):
    """Load policy and world model"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Load policy
    agent = ContextAwareDQN(obs_dim=183, action_dim=4)
    agent.load_state_dict(checkpoint['policy_net'])
    agent.eval()

    # Load world model
    base_path = model_path.replace('_policy.pth', '').replace('_best.pth', '').replace('_final.pth', '')
    world_model_path = f"{base_path}_world_model.pth"

    world_model = None
    wm_type = None

    if os.path.exists(world_model_path):
        wm_checkpoint = torch.load(world_model_path, map_location='cpu', weights_only=False)
        state_dict = wm_checkpoint['model']
        world_model_type = checkpoint.get('world_model_type', 'standard')

        if world_model_type == 'context_aware_fixed':
            # Fixed model
            obs_dim = checkpoint.get('world_model_obs_dim', 180)
            context_dim = checkpoint.get('world_model_context_dim', 3)
            hidden_dim = state_dict['obs_predictor.0.weight'].shape[0] if 'obs_predictor.0.weight' in state_dict else 256

            world_model = ContextAwareWorldModel(
                obs_dim=obs_dim,
                context_dim=context_dim,
                action_dim=4,
                hidden_dim=hidden_dim
            )
            world_model.load_state_dict(state_dict)
            world_model.eval()
            wm_type = 'FIXED'
        else:
            # Standard model
            hidden_dim = state_dict['state_predictor.0.weight'].shape[0] if 'state_predictor.0.weight' in state_dict else 256

            world_model = WorldModelNetwork(state_dim=183, action_dim=4, hidden_dim=hidden_dim)
            world_model.load_state_dict(state_dict)
            world_model.eval()
            wm_type = 'STANDARD'

    return agent, world_model, wm_type, checkpoint


def plan_action(agent, world_model, state, planning_horizon=20):
    """Planning with world model"""
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    best_action = 0
    best_return = -float('inf')

    for action in range(4):
        total_return = 0.0
        for _ in range(5):
            rollout_return = simulate_rollout(agent, world_model, state_tensor, action, planning_horizon)
            total_return += rollout_return
        avg_return = total_return / 5
        if avg_return > best_return:
            best_return = avg_return
            best_action = action

    return best_action


def simulate_rollout(agent, world_model, state, first_action, planning_horizon):
    """Simulate trajectory"""
    current_state = state.clone()
    total_return = 0.0
    discount = 1.0
    gamma = 0.99

    with torch.no_grad():
        action_tensor = torch.LongTensor([first_action])
        next_state, reward, done = world_model(current_state, action_tensor)
        total_return += reward.item() * discount
        discount *= gamma

        if done.item() > 0.5:
            return total_return

        current_state = next_state

        for _ in range(planning_horizon - 1):
            q_values = agent.get_combined_q(current_state)
            action = q_values.argmax(dim=1).item()
            action_tensor = torch.LongTensor([action])
            next_state, reward, done = world_model(current_state, action_tensor)
            total_return += reward.item() * discount
            discount *= gamma
            if done.item() > 0.5:
                break
            current_state = next_state

    return total_return


def test_model(agent, observer, game, world_model, num_episodes, planning_freq=0.2, planning_horizon=20):
    """Test model on game"""
    scores = []
    steps_list = []
    planning_actions = 0
    reactive_actions = 0

    for episode in range(num_episodes):
        game.reset()
        observer.reset()
        game_state = game._get_game_state()

        steps = 0
        done = False

        while not done and steps < 1000:
            obs = observer.observe(game_state)
            context_vector = infer_context_from_observation(obs)
            obs_with_context = add_context_to_observation(obs, context_vector)

            # Action selection
            if world_model and np.random.random() < planning_freq:
                action = plan_action(agent, world_model, obs_with_context, planning_horizon)
                planning_actions += 1
            else:
                action = agent.get_action(obs_with_context, epsilon=0.0)
                reactive_actions += 1

            game_state, reward, done = game.step(action)
            steps += 1

        scores.append(game_state['score'])
        steps_list.append(steps)

    return {
        'avg_score': np.mean(scores),
        'std_score': np.std(scores),
        'max_score': max(scores),
        'min_score': min(scores),
        'avg_steps': np.mean(steps_list),
        'planning_actions': planning_actions,
        'reactive_actions': reactive_actions,
        'scores': scores
    }


def main():
    parser = argparse.ArgumentParser(description='Compare Old vs Fixed Model')
    parser.add_argument('--episodes', type=int, default=30, help='Episodes per game')
    parser.add_argument('--quick', action='store_true', help='Quick test (10 episodes)')
    parser.add_argument('--planning-freq', type=float, default=0.2, help='Planning frequency')
    parser.add_argument('--planning-horizon', type=int, default=20, help='Planning horizon')

    args = parser.parse_args()

    episodes = 10 if args.quick else args.episodes

    print("="*80)
    print("OLD MODEL vs FIXED MODEL COMPARISON")
    print("="*80)
    print(f"Episodes per game: {episodes}")
    print(f"Planning frequency: {args.planning_freq*100:.0f}%")
    print(f"Planning horizon: {args.planning_horizon} steps")
    print("="*80)

    # Model paths
    old_model_path = 'checkpoints/faith_evolution_20251120_091144_final_policy.pth'
    fixed_model_path = 'checkpoints/faith_fixed_20251120_162417_final_policy.pth'

    # Load models
    print("\nLoading OLD model...")
    old_agent, old_wm, old_wm_type, old_checkpoint = load_model_and_world_model(old_model_path)
    old_episodes = len(old_checkpoint.get('episode_rewards', []))
    print(f"  Episodes trained: {old_episodes}")
    print(f"  World model: {old_wm_type or 'None'}")

    print("\nLoading FIXED model...")
    fixed_agent, fixed_wm, fixed_wm_type, fixed_checkpoint = load_model_and_world_model(fixed_model_path)
    fixed_episodes = len(fixed_checkpoint.get('episode_rewards', []))
    print(f"  Episodes trained: {fixed_episodes}")
    print(f"  World model: {fixed_wm_type or 'None'}")

    # Create observer
    observer = ExpandedTemporalObserver(num_rays=16, ray_length=15, verbose=False)

    # Test on games
    games = {
        'snake': SnakeGame(size=20),
        'pacman': PacManGame(size=20),
        'dungeon': DungeonGame(size=20)
    }

    old_results = {}
    fixed_results = {}

    for game_name, game in games.items():
        print(f"\n{'='*80}")
        print(f"TESTING: {game_name.upper()}")
        print(f"{'='*80}")

        print(f"\nOLD model...")
        old_results[game_name] = test_model(
            old_agent, observer, game, old_wm,
            episodes, args.planning_freq, args.planning_horizon
        )
        print(f"  Avg score: {old_results[game_name]['avg_score']:.2f}")

        print(f"\nFIXED model...")
        fixed_results[game_name] = test_model(
            fixed_agent, observer, game, fixed_wm,
            episodes, args.planning_freq, args.planning_horizon
        )
        print(f"  Avg score: {fixed_results[game_name]['avg_score']:.2f}")

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"DETAILED COMPARISON")
    print(f"{'='*80}")

    print(f"\n{'Game':<15} {'Metric':<20} {'OLD':<20} {'FIXED':<20} {'Δ':<15}")
    print(f"{'-'*80}")

    for game_name in games.keys():
        old = old_results[game_name]
        fixed = fixed_results[game_name]

        delta_score = fixed['avg_score'] - old['avg_score']
        pct_change = (delta_score / old['avg_score'] * 100) if old['avg_score'] != 0 else 0

        print(f"{game_name.capitalize():<15} {'Avg Score':<20} "
              f"{old['avg_score']:>7.2f} ± {old['std_score']:<5.2f}  "
              f"{fixed['avg_score']:>7.2f} ± {fixed['std_score']:<5.2f}  "
              f"{delta_score:>+6.2f} ({pct_change:>+5.1f}%)")

        print(f"{'':15} {'Max Score':<20} "
              f"{old['max_score']:>7}             "
              f"{fixed['max_score']:>7}             "
              f"{fixed['max_score'] - old['max_score']:>+6}")

        print(f"{'':15} {'Min Score':<20} "
              f"{old['min_score']:>7}             "
              f"{fixed['min_score']:>7}             "
              f"{fixed['min_score'] - old['min_score']:>+6}")

        print(f"{'':15} {'Avg Steps':<20} "
              f"{old['avg_steps']:>7.1f}             "
              f"{fixed['avg_steps']:>7.1f}             "
              f"{fixed['avg_steps'] - old['avg_steps']:>+6.1f}")

        # Planning usage
        old_total = old['planning_actions'] + old['reactive_actions']
        fixed_total = fixed['planning_actions'] + fixed['reactive_actions']
        old_planning_pct = old['planning_actions'] / old_total * 100 if old_total > 0 else 0
        fixed_planning_pct = fixed['planning_actions'] / fixed_total * 100 if fixed_total > 0 else 0

        print(f"{'':15} {'Planning Usage':<20} "
              f"{old_planning_pct:>7.1f}%            "
              f"{fixed_planning_pct:>7.1f}%            "
              f"{fixed_planning_pct - old_planning_pct:>+6.1f}%")

        print()

    # Overall summary
    print(f"{'='*80}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*80}")

    old_total = sum(r['avg_score'] for r in old_results.values())
    fixed_total = sum(r['avg_score'] for r in fixed_results.values())
    overall_improvement = ((fixed_total - old_total) / old_total * 100) if old_total != 0 else 0

    print(f"\nTotal Score (all games):")
    print(f"  OLD model:   {old_total:.2f}")
    print(f"  FIXED model: {fixed_total:.2f}")
    print(f"  Improvement: {fixed_total - old_total:>+.2f} ({overall_improvement:>+.1f}%)")

    # Verdict
    print(f"\n{'='*80}")
    print(f"VERDICT")
    print(f"{'='*80}")

    if overall_improvement > 10:
        print(f"\n✅ FIXED model shows SIGNIFICANT improvement ({overall_improvement:+.1f}%)")
        print(f"   The bottleneck fix had major positive impact!")
        print(f"\n   Recommendation: Use FIXED model for all applications")
    elif overall_improvement > 0:
        print(f"\n✅ FIXED model shows improvement ({overall_improvement:+.1f}%)")
        print(f"   The bottleneck fix helped performance")
        print(f"\n   Recommendation: Use FIXED model")
    elif overall_improvement > -5:
        print(f"\n⚠️  Models perform similarly ({overall_improvement:+.1f}%)")
        print(f"   Fix didn't hurt, but benefit is small")
        print(f"\n   Recommendation: Use FIXED model (cleaner architecture)")
    else:
        print(f"\n⚠️  OLD model performs better ({overall_improvement:+.1f}%)")
        print(f"   Unexpected - may need investigation")

    # Game-by-game breakdown
    print(f"\n\nGAME-BY-GAME ANALYSIS:")
    for game_name in games.keys():
        old = old_results[game_name]
        fixed = fixed_results[game_name]
        delta = fixed['avg_score'] - old['avg_score']
        pct = (delta / old['avg_score'] * 100) if old['avg_score'] != 0 else 0

        if pct > 10:
            status = "✅ Major improvement"
        elif pct > 0:
            status = "✅ Improved"
        elif pct > -5:
            status = "≈ Similar"
        else:
            status = "⚠️ Regressed"

        print(f"  {game_name.capitalize():8s}: {delta:>+6.2f} ({pct:>+5.1f}%) - {status}")

    print("="*80)


if __name__ == '__main__':
    main()
