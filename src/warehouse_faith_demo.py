"""
Warehouse Faith Scenarios - Demonstration with Updated Architecture

Tests trained agent (baseline or expanded) on realistic warehouse scenarios.
Supports both architecture types:
- BASELINE: 8 rays × 10 tiles (95 dims total)
- EXPANDED: 16 rays × 15 tiles (183 dims total)

Demonstrates:
1. Hidden mechanic discovery via faith-based exploration
2. Entity recognition and behavior learning
3. Pattern transfer across scenarios
4. Adaptive decision-making

Usage:
    python warehouse_faith_demo.py checkpoints/faith_evolution_20251120_091144_final_policy.pth
    python warehouse_faith_demo.py <model_path> --scenario hidden_shortcut
    python warehouse_faith_demo.py <model_path> --scenario all --episodes 20
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))

import torch
import numpy as np
import argparse
from collections import defaultdict

# Warehouse scenarios
from warehouse_faith_scenarios import (
    HiddenShortcutScenario,
    ChargingStationScenario,
    PriorityZoneScenario
)

# Agent and observers
from context_aware_agent import (
    ContextAwareDQN,
    infer_context_from_observation,
    add_context_to_observation
)
from core.temporal_observer import TemporalFlowObserver  # Baseline
from core.expanded_temporal_observer import ExpandedTemporalObserver  # Expanded

# Revolutionary systems
from core.faith_system import FaithPopulation
from core.entity_discovery import EntityDiscoveryWorldModel
from core.world_model import WorldModelNetwork


def detect_model_architecture(checkpoint):
    """Detect if model uses baseline or expanded architecture"""
    state_dict = checkpoint['policy_net']

    # Check first layer size to determine architecture
    # Baseline: 95 input dims, Expanded: 183 input dims

    # Try different possible first layer keys (model architecture may vary)
    possible_keys = [
        'perception_net.0.weight',  # ContextAwareDQN with perception net
        'q_heads.snake.0.weight',   # ContextAwareDQN with direct q_heads
        'fc1.weight',               # Simple DQN
    ]

    for key in possible_keys:
        if key in state_dict:
            # Shape is [hidden_dim, input_dim]
            input_dim = state_dict[key].shape[1]

            if input_dim == 95:
                return 'baseline', 95
            elif input_dim == 183:
                return 'expanded', 183
            else:
                # Unknown architecture, but we have the dimension
                print(f"  Warning: Unexpected input dimension {input_dim}")
                print(f"  Proceeding with detected dimension...")

                # Guess architecture based on dimension
                if input_dim < 100:
                    return 'baseline', input_dim
                else:
                    return 'expanded', input_dim

    # Could not detect - try to infer from keys
    print(f"  Warning: Could not find standard layer keys")
    print(f"  Available keys: {list(state_dict.keys())[:10]}")
    return 'unknown', None


def create_observer_for_architecture(arch_type):
    """Create appropriate observer for model architecture"""
    if arch_type == 'baseline':
        return TemporalFlowObserver(num_rays=8, ray_length=10)  # 92 dims
    elif arch_type == 'expanded':
        return ExpandedTemporalObserver(num_rays=16, ray_length=15, verbose=False)  # 180 dims
    else:
        raise ValueError(f"Unknown architecture type: {arch_type}")


def test_scenario(agent, observer, scenario, scenario_name, num_episodes=20,
                  world_model=None, faith_population=None,
                  planning_freq=0.2, faith_freq=0.3):
    """
    Test agent on warehouse scenario.

    Args:
        agent: Policy network
        observer: Temporal observer (baseline or expanded)
        scenario: Warehouse scenario instance
        scenario_name: Scenario name
        num_episodes: Number of test episodes
        world_model: Optional world model for planning
        faith_population: Optional faith pattern population
        planning_freq: Planning action frequency
        faith_freq: Faith action frequency
    """
    print(f"\n{'='*80}")
    print(f"TESTING: {scenario.name}")
    print(f"{'='*80}")
    print(f"Description: {scenario.description}")
    print(f"Episodes: {num_episodes}")
    print()

    # Tracking
    episode_scores = []
    episode_steps = []
    mechanic_discoveries = []
    faith_actions = 0
    planning_actions = 0
    reactive_actions = 0

    # Scenario-specific tracking
    shortcut_uses_total = 0
    optimal_charges_total = 0
    priority_discoveries_total = 0

    for episode in range(num_episodes):
        state = scenario.reset()
        observer.reset()

        # Select faith pattern for episode
        active_faith_pattern = None
        if faith_population:
            active_faith_pattern = faith_population.select_pattern_for_episode()

        episode_reward = 0
        done = False

        while not done:
            # Get observation from scenario state
            obs = observer.observe(state)

            # Infer context
            context_vector = infer_context_from_observation(obs)

            # Add context
            obs_with_context = add_context_to_observation(obs, context_vector)

            # ACTION SELECTION: Faith → Planning → Reactive
            action_source = 'reactive'

            # Option 1: Faith-based action
            if active_faith_pattern and np.random.random() < faith_freq:
                obs_tensor = torch.FloatTensor(obs_with_context).unsqueeze(0)
                base_action = agent.get_action(obs_with_context, epsilon=0.0)

                if active_faith_pattern.should_override(scenario.steps, obs_tensor):
                    action = active_faith_pattern.get_action(obs_tensor, base_action, scenario.steps)
                    action_source = 'faith'
                    faith_actions += 1
                else:
                    action = base_action
                    reactive_actions += 1

            # Option 2: Planning action
            elif world_model and np.random.random() < planning_freq:
                # Simple planning: try each action, pick best
                action = _simple_plan(agent, world_model, obs_with_context)
                action_source = 'planning'
                planning_actions += 1

            # Option 3: Reactive action
            else:
                action = agent.get_action(obs_with_context, epsilon=0.0)
                reactive_actions += 1

            # Execute action
            state, reward, done = scenario.step(action)
            episode_reward += reward

        # Record episode results
        episode_scores.append(scenario.packages_picked)
        episode_steps.append(scenario.steps)

        # Record discoveries
        if scenario.hidden_mechanics_discovered:
            for discovery in scenario.hidden_mechanics_discovered:
                if discovery not in mechanic_discoveries:
                    mechanic_discoveries.append(discovery)

        # Scenario-specific tracking
        scenario_info = scenario.get_scenario_info()
        if 'shortcut_uses' in scenario_info:
            shortcut_uses_total += scenario_info['shortcut_uses']
        if 'optimal_charges' in scenario_info:
            optimal_charges_total += scenario_info['optimal_charges']
        if 'strategy_found' in scenario_info and scenario_info['strategy_found']:
            priority_discoveries_total += 1

    # Print results
    print(f"RESULTS:")
    print(f"  Average packages collected: {np.mean(episode_scores):.2f} ± {np.std(episode_scores):.2f}")
    print(f"  Max packages: {max(episode_scores)}")
    print(f"  Average steps: {np.mean(episode_steps):.1f}")
    print(f"  Average collisions per episode: {np.mean([scenario.collisions]) if hasattr(scenario, 'collisions') else 0:.2f}")

    print(f"\nACTION DISTRIBUTION:")
    total_actions = faith_actions + planning_actions + reactive_actions
    if total_actions > 0:
        print(f"  Faith:    {faith_actions/total_actions*100:>5.1f}% ({faith_actions:>6} actions)")
        print(f"  Planning: {planning_actions/total_actions*100:>5.1f}% ({planning_actions:>6} actions)")
        print(f"  Reactive: {reactive_actions/total_actions*100:>5.1f}% ({reactive_actions:>6} actions)")

    print(f"\nHIDDEN MECHANIC DISCOVERIES:")
    if mechanic_discoveries:
        print(f"  Total mechanics discovered: {len(mechanic_discoveries)}")
        for discovery in mechanic_discoveries:
            print(f"    - {discovery['mechanic']}: {discovery['description']}")
            print(f"      Discovered at step: {discovery['step']}")
    else:
        print(f"  No hidden mechanics discovered (need more exploration)")

    # Scenario-specific results
    if shortcut_uses_total > 0:
        print(f"\nSHORTCUT USAGE:")
        print(f"  Total shortcut uses: {shortcut_uses_total}")
        print(f"  Average per episode: {shortcut_uses_total/num_episodes:.2f}")

    if optimal_charges_total > 0:
        print(f"\nBATTERY MANAGEMENT:")
        print(f"  Optimal charging sessions: {optimal_charges_total}")
        print(f"  Average per episode: {optimal_charges_total/num_episodes:.2f}")

    if priority_discoveries_total > 0:
        print(f"\nPRIORITY STRATEGY:")
        print(f"  Episodes with optimal strategy: {priority_discoveries_total}/{num_episodes}")

    return {
        'scores': episode_scores,
        'steps': episode_steps,
        'avg_score': np.mean(episode_scores),
        'discoveries': mechanic_discoveries,
        'action_distribution': {
            'faith': faith_actions,
            'planning': planning_actions,
            'reactive': reactive_actions
        }
    }


def _simple_plan(agent, world_model, state):
    """Simple 1-step lookahead planning"""
    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    best_action = 0
    best_value = -float('inf')

    with torch.no_grad():
        for action in range(4):
            action_tensor = torch.LongTensor([action])
            next_state, reward, done = world_model(state_tensor, action_tensor)

            # Evaluate next state
            q_values = agent.get_combined_q(next_state)
            value = reward.item() + 0.99 * q_values.max().item()

            if value > best_value:
                best_value = value
                best_action = action

    return best_action


def main():
    parser = argparse.ArgumentParser(description='Warehouse Faith Scenarios Demo')
    parser.add_argument('model_path', help='Path to trained policy model')
    parser.add_argument('--scenario', choices=['hidden_shortcut', 'charging_station',
                                               'priority_zone', 'all'], default='all')
    parser.add_argument('--episodes', type=int, default=20, help='Episodes per scenario')
    parser.add_argument('--no-faith', action='store_true', help='Disable faith actions')
    parser.add_argument('--faith-freq', type=float, default=0.3, help='Faith action frequency')
    parser.add_argument('--planning-freq', type=float, default=0.2, help='Planning frequency')

    args = parser.parse_args()

    print(f"{'='*80}")
    print(f"WAREHOUSE FAITH SCENARIOS DEMONSTRATION")
    print(f"{'='*80}")
    print(f"Model: {args.model_path}")
    print(f"Episodes per scenario: {args.episodes}")

    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)

    # Detect architecture
    arch_type, obs_dim = detect_model_architecture(checkpoint)

    print(f"\nMODEL ARCHITECTURE DETECTED:")
    if arch_type == 'baseline':
        print(f"  Type: BASELINE (8 rays × 10 tiles)")
        print(f"  Observation dims: {obs_dim} (92 + 3 context)")
    elif arch_type == 'expanded':
        print(f"  Type: EXPANDED (16 rays × 15 tiles)")
        print(f"  Observation dims: {obs_dim} (180 + 3 context)")
        print(f"  Multi-scale temporal: Enabled")
    else:
        print(f"  Type: UNKNOWN (obs_dim={obs_dim})")

    # Load agent
    agent = ContextAwareDQN(obs_dim=obs_dim, action_dim=4)
    agent.load_state_dict(checkpoint['policy_net'])
    agent.eval()

    print(f"\nTRAINING INFO:")
    print(f"  Episodes trained: {len(checkpoint.get('episode_rewards', []))}")
    print(f"  Steps: {checkpoint.get('steps_done', 0):,}")

    # Create observer
    observer = create_observer_for_architecture(arch_type)
    print(f"\nOBSERVER:")
    print(f"  Type: {arch_type}")
    print(f"  Observation dims: {observer.obs_dim}")

    # Load world model
    world_model = None
    base_path = args.model_path.replace('_policy.pth', '').replace('_best.pth', '').replace('_final.pth', '')
    world_model_path = f"{base_path}_world_model.pth"

    if os.path.exists(world_model_path):
        wm_checkpoint = torch.load(world_model_path, map_location='cpu', weights_only=False)
        state_dict = wm_checkpoint['model']

        # Check for FIXED world model
        world_model_type = checkpoint.get('world_model_type', 'standard')

        if world_model_type == 'context_aware_fixed':
            # FIXED world model
            from core.context_aware_world_model import ContextAwareWorldModel

            obs_dim_only = checkpoint.get('world_model_obs_dim', 180)
            context_dim = checkpoint.get('world_model_context_dim', 3)

            if 'obs_predictor.0.weight' in state_dict:
                hidden_dim = state_dict['obs_predictor.0.weight'].shape[0]
            else:
                hidden_dim = 256

            world_model = ContextAwareWorldModel(
                obs_dim=obs_dim_only,
                context_dim=context_dim,
                action_dim=4,
                hidden_dim=hidden_dim
            )
            world_model.load_state_dict(state_dict)
            world_model.eval()
            print(f"\nWORLD MODEL:")
            print(f"  Type: FIXED Context-Aware (bottleneck removed)")
            print(f"  Loaded: Yes (obs={obs_dim_only}, hidden={hidden_dim})")
        else:
            # Standard world model
            if 'state_predictor.0.weight' in state_dict:
                hidden_dim = state_dict['state_predictor.0.weight'].shape[0]
            else:
                hidden_dim = 256

            world_model = WorldModelNetwork(state_dim=obs_dim, action_dim=4, hidden_dim=hidden_dim)
            world_model.load_state_dict(state_dict)
            world_model.eval()
            print(f"\nWORLD MODEL:")
            print(f"  Type: Standard")
            print(f"  Loaded: Yes (hidden_dim={hidden_dim})")

        print(f"  Planning frequency: {args.planning_freq*100:.0f}%")
    else:
        print(f"\nWORLD MODEL: Not found")

    # Load faith population
    faith_population = None
    if not args.no_faith:
        faith_population = FaithPopulation(population_size=20, signature_dim=32)
        print(f"\nFAITH SYSTEM:")
        print(f"  Population: 20 patterns")
        print(f"  Frequency: {args.faith_freq*100:.0f}%")

    # Test scenarios
    scenarios_to_test = {
        'hidden_shortcut': HiddenShortcutScenario,
        'charging_station': ChargingStationScenario,
        'priority_zone': PriorityZoneScenario
    }

    if args.scenario != 'all':
        scenarios_to_test = {args.scenario: scenarios_to_test[args.scenario]}

    results = {}

    for scenario_name, scenario_class in scenarios_to_test.items():
        scenario = scenario_class(size=20)
        results[scenario_name] = test_scenario(
            agent=agent,
            observer=observer,
            scenario=scenario,
            scenario_name=scenario_name,
            num_episodes=args.episodes,
            world_model=world_model,
            faith_population=faith_population,
            planning_freq=args.planning_freq,
            faith_freq=args.faith_freq
        )

    # Overall summary
    if len(results) > 1:
        print(f"\n{'='*80}")
        print(f"OVERALL SUMMARY")
        print(f"{'='*80}")

        total_discoveries = sum(len(r['discoveries']) for r in results.values())
        print(f"\nTotal mechanic discoveries: {total_discoveries}")

        print(f"\nScenario Performance:")
        for name, result in results.items():
            print(f"  {name:20s}: {result['avg_score']:>6.2f} avg packages")

        # Action effectiveness
        print(f"\nAction Distribution (Overall):")
        total_faith = sum(r['action_distribution']['faith'] for r in results.values())
        total_planning = sum(r['action_distribution']['planning'] for r in results.values())
        total_reactive = sum(r['action_distribution']['reactive'] for r in results.values())
        total_actions = total_faith + total_planning + total_reactive

        if total_actions > 0:
            print(f"  Faith:    {total_faith/total_actions*100:>5.1f}%")
            print(f"  Planning: {total_planning/total_actions*100:>5.1f}%")
            print(f"  Reactive: {total_reactive/total_actions*100:>5.1f}%")


if __name__ == '__main__':
    main()
