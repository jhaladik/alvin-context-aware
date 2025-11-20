"""
Comprehensive Configuration Testing - Find Optimal Setup

Tests multiple configurations to determine best setup:
1. Faith frequency variations (0%, 10%, 20%, 30%, 40%)
2. Planning frequency variations (0%, 10%, 20%, 30%)
3. Planning horizon variations (5, 10, 20, 30 steps)
4. Model comparisons (old vs fixed)

Generates detailed comparison report showing which configuration performs best.

Usage:
    python run_configuration_tests.py --model checkpoints/faith_fixed_20251120_162417_final_policy.pth
    python run_configuration_tests.py --quick  # Quick test (fewer configs, fewer episodes)
    python run_configuration_tests.py --full   # Full test (all configs, many episodes)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))

import torch
import numpy as np
import argparse
from datetime import datetime
import json

from context_aware_agent import ContextAwareDQN, infer_context_from_observation, add_context_to_observation
from core.planning_test_games import SnakeGame, PacManGame, DungeonGame
from core.faith_system import FaithPopulation


def detect_model_architecture(checkpoint):
    """Detect model architecture"""
    state_dict = checkpoint['policy_net']
    possible_keys = ['perception_net.0.weight', 'q_heads.snake.0.weight', 'fc1.weight']

    for key in possible_keys:
        if key in state_dict:
            input_dim = state_dict[key].shape[1]
            if input_dim == 95:
                return 'baseline', 95, 92
            elif input_dim == 183:
                return 'expanded', 183, 180
            else:
                return 'unknown', input_dim, input_dim - 3
    return 'unknown', None, None


def create_observer(arch_type):
    """Create appropriate observer"""
    if arch_type == 'baseline':
        from core.temporal_observer import TemporalFlowObserver
        return TemporalFlowObserver(num_rays=8, ray_length=10)
    else:
        from core.expanded_temporal_observer import ExpandedTemporalObserver
        return ExpandedTemporalObserver(num_rays=16, ray_length=15, verbose=False)


def load_world_model(model_path, checkpoint, obs_dim_total, obs_dim_only):
    """Load world model with architecture detection"""
    base_path = model_path.replace('_policy.pth', '').replace('_best.pth', '').replace('_final.pth', '')
    world_model_path = f"{base_path}_world_model.pth"

    if not os.path.exists(world_model_path):
        return None

    try:
        wm_checkpoint = torch.load(world_model_path, map_location='cpu', weights_only=False)
        state_dict = wm_checkpoint['model']
        world_model_type = checkpoint.get('world_model_type', 'standard')

        if world_model_type == 'context_aware_fixed':
            from core.context_aware_world_model import ContextAwareWorldModel

            obs_dim_wm = checkpoint.get('world_model_obs_dim', obs_dim_only)
            context_dim = checkpoint.get('world_model_context_dim', 3)
            hidden_dim = state_dict['obs_predictor.0.weight'].shape[0] if 'obs_predictor.0.weight' in state_dict else 256

            world_model = ContextAwareWorldModel(
                obs_dim=obs_dim_wm,
                context_dim=context_dim,
                action_dim=4,
                hidden_dim=hidden_dim
            )
            world_model.load_state_dict(state_dict)
            world_model.eval()
            return world_model, 'fixed'
        else:
            from core.world_model import WorldModelNetwork
            hidden_dim = state_dict['state_predictor.0.weight'].shape[0] if 'state_predictor.0.weight' in state_dict else 256

            world_model = WorldModelNetwork(state_dim=obs_dim_total, action_dim=4, hidden_dim=hidden_dim)
            world_model.load_state_dict(state_dict)
            world_model.eval()
            return world_model, 'standard'
    except Exception as e:
        print(f"Warning: Could not load world model: {e}")
        return None


def plan_action(agent, world_model, state, planning_horizon):
    """Planning with world model"""
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    best_action = 0
    best_return = -float('inf')

    for action in range(4):
        total_return = 0.0
        for _ in range(5):  # 5 rollouts
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


def test_configuration(agent, observer, game, world_model, faith_population,
                       config, num_episodes=20):
    """Test single configuration"""
    scores = []
    steps_list = []
    action_counts = {'faith': 0, 'planning': 0, 'reactive': 0}

    for episode in range(num_episodes):
        game.reset()
        observer.reset()
        game_state = game._get_game_state()

        total_reward = 0
        steps = 0
        done = False

        while not done and steps < 1000:
            obs = observer.observe(game_state)
            context_vector = infer_context_from_observation(obs)
            obs_with_context = add_context_to_observation(obs, context_vector)

            # Action selection based on configuration
            rand = np.random.random()
            if rand < config['faith_freq']:
                action = np.random.randint(4)  # Simplified faith action
                action_counts['faith'] += 1
            elif rand < config['faith_freq'] + config['planning_freq'] and world_model is not None:
                action = plan_action(agent, world_model, obs_with_context, config['planning_horizon'])
                action_counts['planning'] += 1
            else:
                action = agent.get_action(obs_with_context, epsilon=0.0)
                action_counts['reactive'] += 1

            game_state, reward, done = game.step(action)
            total_reward += reward
            steps += 1

        scores.append(game_state['score'])
        steps_list.append(steps)

    return {
        'avg_score': np.mean(scores),
        'std_score': np.std(scores),
        'max_score': max(scores),
        'avg_steps': np.mean(steps_list),
        'action_counts': action_counts,
        'scores': scores
    }


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Configuration Testing')
    parser.add_argument('--model', type=str,
                       default='checkpoints/faith_fixed_20251120_162417_final_policy.pth',
                       help='Model to test')
    parser.add_argument('--quick', action='store_true', help='Quick test (fewer configs)')
    parser.add_argument('--full', action='store_true', help='Full test (all configs)')
    parser.add_argument('--episodes', type=int, default=None, help='Episodes per config')
    parser.add_argument('--output', type=str, default='config_test_results.json',
                       help='Output file for results')

    args = parser.parse_args()

    # Determine test scope
    if args.quick:
        episodes_per_config = args.episodes or 10
        faith_freqs = [0.0, 0.3]
        planning_freqs = [0.0, 0.2]
        planning_horizons = [20]
        games_to_test = ['pacman']
    elif args.full:
        episodes_per_config = args.episodes or 30
        faith_freqs = [0.0, 0.1, 0.2, 0.3, 0.4]
        planning_freqs = [0.0, 0.1, 0.2, 0.3]
        planning_horizons = [5, 10, 20, 30]
        games_to_test = ['snake', 'pacman', 'dungeon']
    else:
        episodes_per_config = args.episodes or 20
        faith_freqs = [0.0, 0.2, 0.3]
        planning_freqs = [0.0, 0.2]
        planning_horizons = [10, 20]
        games_to_test = ['pacman']

    print("="*80)
    print("COMPREHENSIVE CONFIGURATION TESTING")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Episodes per config: {episodes_per_config}")
    print(f"Faith frequencies: {faith_freqs}")
    print(f"Planning frequencies: {planning_freqs}")
    print(f"Planning horizons: {planning_horizons}")
    print(f"Games: {games_to_test}")
    print("="*80)

    # Load model
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    arch_type, obs_dim_total, obs_dim_only = detect_model_architecture(checkpoint)

    print(f"\nModel Architecture: {arch_type}")
    print(f"  Observation dims: {obs_dim_total}")

    agent = ContextAwareDQN(obs_dim=obs_dim_total, action_dim=4)
    agent.load_state_dict(checkpoint['policy_net'])
    agent.eval()

    observer = create_observer(arch_type)
    world_model_result = load_world_model(args.model, checkpoint, obs_dim_total, obs_dim_only)

    if world_model_result:
        world_model, wm_type = world_model_result
        print(f"  World model: {wm_type}")
    else:
        world_model = None
        print(f"  World model: None (planning disabled)")

    faith_population = FaithPopulation(population_size=20)

    # Generate configurations to test
    configurations = []
    for faith_freq in faith_freqs:
        for planning_freq in planning_freqs:
            for planning_horizon in planning_horizons:
                if world_model is None and planning_freq > 0:
                    continue  # Skip planning configs if no world model

                configurations.append({
                    'faith_freq': faith_freq,
                    'planning_freq': planning_freq,
                    'planning_horizon': planning_horizon,
                    'name': f"F{int(faith_freq*100)}_P{int(planning_freq*100)}_H{planning_horizon}"
                })

    print(f"\nTotal configurations to test: {len(configurations)}")
    print()

    # Test all configurations
    all_results = {}

    for game_name in games_to_test:
        print(f"\n{'='*80}")
        print(f"TESTING GAME: {game_name.upper()}")
        print(f"{'='*80}")

        if game_name == 'snake':
            game = SnakeGame(size=20)
        elif game_name == 'pacman':
            game = PacManGame(size=20)
        else:
            game = DungeonGame(size=20)

        game_results = []

        for i, config in enumerate(configurations):
            print(f"\n[{i+1}/{len(configurations)}] Testing: {config['name']}")
            print(f"  Faith: {config['faith_freq']*100:.0f}%, "
                  f"Planning: {config['planning_freq']*100:.0f}%, "
                  f"Horizon: {config['planning_horizon']}")

            result = test_configuration(
                agent, observer, game, world_model, faith_population,
                config, num_episodes=episodes_per_config
            )

            result['config'] = config
            game_results.append(result)

            print(f"  Result: Avg={result['avg_score']:.2f}, "
                  f"Max={result['max_score']}, "
                  f"Steps={result['avg_steps']:.1f}")

            # Print action distribution
            total_actions = sum(result['action_counts'].values())
            if total_actions > 0:
                print(f"  Actions: F={result['action_counts']['faith']/total_actions*100:.1f}%, "
                      f"P={result['action_counts']['planning']/total_actions*100:.1f}%, "
                      f"R={result['action_counts']['reactive']/total_actions*100:.1f}%")

        all_results[game_name] = game_results

        # Find best configuration for this game
        best_config = max(game_results, key=lambda x: x['avg_score'])
        print(f"\n{'-'*80}")
        print(f"BEST CONFIGURATION FOR {game_name.upper()}:")
        print(f"  {best_config['config']['name']}")
        print(f"  Faith: {best_config['config']['faith_freq']*100:.0f}%, "
              f"Planning: {best_config['config']['planning_freq']*100:.0f}%, "
              f"Horizon: {best_config['config']['planning_horizon']}")
        print(f"  Score: {best_config['avg_score']:.2f} ± {best_config['std_score']:.2f}")
        print(f"  Max: {best_config['max_score']}")

    # Overall analysis
    print(f"\n{'='*80}")
    print(f"OVERALL ANALYSIS")
    print(f"{'='*80}")

    # Find best overall configuration (average across games)
    config_scores = {}
    for config in configurations:
        total_score = 0
        count = 0
        for game_name in games_to_test:
            for result in all_results[game_name]:
                if result['config']['name'] == config['name']:
                    total_score += result['avg_score']
                    count += 1
        config_scores[config['name']] = total_score / count if count > 0 else 0

    best_overall = max(config_scores.items(), key=lambda x: x[1])

    print(f"\nBEST OVERALL CONFIGURATION:")
    print(f"  {best_overall[0]}")
    print(f"  Average score across all games: {best_overall[1]:.2f}")

    # Parse config name to get parameters
    parts = best_overall[0].split('_')
    best_faith = int(parts[0][1:]) / 100
    best_planning = int(parts[1][1:]) / 100
    best_horizon = int(parts[2][1:])

    print(f"  Faith frequency: {best_faith*100:.0f}%")
    print(f"  Planning frequency: {best_planning*100:.0f}%")
    print(f"  Planning horizon: {best_horizon} steps")

    # Show top 5 configurations
    print(f"\nTOP 5 CONFIGURATIONS:")
    top_5 = sorted(config_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (config_name, score) in enumerate(top_5, 1):
        print(f"  {i}. {config_name}: {score:.2f}")

    # Save results
    output_data = {
        'model': args.model,
        'architecture': arch_type,
        'world_model_type': wm_type if world_model_result else None,
        'episodes_per_config': episodes_per_config,
        'test_date': datetime.now().isoformat(),
        'results': all_results,
        'config_scores': config_scores,
        'best_overall': {
            'name': best_overall[0],
            'score': best_overall[1],
            'faith_freq': best_faith,
            'planning_freq': best_planning,
            'planning_horizon': best_horizon
        }
    }

    # Convert numpy types to native Python for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj

    output_data = convert_to_native(output_data)

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {args.output}")

    # Print recommendation
    print(f"\n{'='*80}")
    print(f"RECOMMENDATION")
    print(f"{'='*80}")
    print(f"\nFor best performance, use:")
    print(f"  --faith-freq {best_faith}")
    print(f"  --planning-freq {best_planning}")
    print(f"  --planning-horizon {best_horizon}")
    print()
    print(f"Example command:")
    print(f"  python demo_pacman_faith_expanded.py \\")
    print(f"    --model {args.model} \\")
    print(f"    --faith-freq {best_faith} \\")
    print(f"    --planning-freq {best_planning} \\")
    print(f"    --planning-horizon {best_horizon}")
    print("="*80)


if __name__ == '__main__':
    main()
