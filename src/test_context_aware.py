"""
Test Context-Aware Agent on Snake, Pac-Man, and Dungeon.

Tests agent's ability to adapt behavior based on context:
- Snake (0 entities): Should focus on collection
- Balanced (2-3 entities): Should balance collection and avoidance
- Survival (4+ entities): Should prioritize survival

Usage:
    python test_context_aware.py checkpoints/context_aware_20251117_220042_best_policy.pth
    python test_context_aware.py <model_path> --episodes 100
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))

import torch
import numpy as np
import argparse
from context_aware_agent import (
    ContextAwareDQN,
    infer_context_from_observation,
    add_context_to_observation
)
from core.temporal_observer import TemporalFlowObserver
from core.planning_test_games import SnakeGame, PacManGame, DungeonGame
from core.world_model import WorldModelNetwork


def _plan_action(agent, world_model, state, planning_horizon=5):
    """Use world model to plan best action via lookahead"""
    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    best_action = None
    best_return = -float('inf')

    # Try each action
    for action in range(4):  # 4 actions: UP, DOWN, LEFT, RIGHT
        total_return = 0.0

        # Monte Carlo: simulate multiple rollouts
        num_rollouts = 5
        for _ in range(num_rollouts):
            rollout_return = _simulate_rollout(agent, world_model, state_tensor, action, planning_horizon)
            total_return += rollout_return

        avg_return = total_return / num_rollouts

        if avg_return > best_return:
            best_return = avg_return
            best_action = action

    return best_action


def _simulate_rollout(agent, world_model, state, first_action, planning_horizon=5):
    """Simulate one trajectory using world model"""
    current_state = state.clone()
    total_return = 0.0
    discount = 1.0
    gamma = 0.99

    with torch.no_grad():
        # Take first action
        action_tensor = torch.LongTensor([first_action])
        next_state, reward, done = world_model(current_state, action_tensor)
        total_return += reward.item() * discount
        discount *= gamma

        if done.item() > 0.5:
            return total_return

        current_state = next_state

        # Simulate remaining horizon steps using policy
        for _ in range(planning_horizon - 1):
            # Use policy to select action
            q_values = agent.get_combined_q(current_state)
            action = q_values.argmax(dim=1).item()

            # Simulate with world model
            action_tensor = torch.LongTensor([action])
            next_state, reward, done = world_model(current_state, action_tensor)

            total_return += reward.item() * discount
            discount *= gamma

            if done.item() > 0.5:
                break

            current_state = next_state

    return total_return


def test_game(agent, observer, game, game_name, num_episodes=50, world_model=None, planning_freq=0.3, planning_horizon=5):
    """Test agent on a specific game"""
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

            # Infer context from observation
            context_vector = infer_context_from_observation(obs)

            # Track context distribution
            if context_vector[0] == 1.0:
                context_counts['snake'] += 1
            elif context_vector[1] == 1.0:
                context_counts['balanced'] += 1
            else:
                context_counts['survival'] += 1

            # Add context to observation
            obs_with_context = add_context_to_observation(obs, context_vector)

            # Get action (with planning if available)
            if world_model is not None and np.random.random() < planning_freq:
                action = _plan_action(agent, world_model, obs_with_context, planning_horizon)
            else:
                action = agent.get_action(obs_with_context, epsilon=0.0)

            # Execute action
            game_state, reward, done = game.step(action)
            total_reward += reward
            steps += 1

        scores.append(game_state['score'])
        steps_list.append(steps)

    # Print results
    print(f"\n{'='*60}")
    print(f"{game_name.upper()} TEST RESULTS")
    print(f"{'='*60}")
    print(f"Episodes: {num_episodes}")
    print(f"Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Max Score: {max(scores)}")
    print(f"Min Score: {min(scores)}")
    print(f"Average Steps: {np.mean(steps_list):.1f}")
    print()
    print("Context Distribution:")
    total_steps = sum(context_counts.values())
    for context, count in context_counts.items():
        pct = (count / total_steps * 100) if total_steps > 0 else 0
        print(f"  {context:8s}: {count:5d} steps ({pct:5.1f}%)")

    return {
        'scores': scores,
        'steps': steps_list,
        'avg_score': np.mean(scores),
        'context_counts': context_counts
    }


def main():
    parser = argparse.ArgumentParser(description='Test Context-Aware Agent')
    parser.add_argument('model_path', help='Path to policy checkpoint')
    parser.add_argument('--episodes', type=int, default=50, help='Episodes per game')
    parser.add_argument('--game', choices=['snake', 'pacman', 'dungeon', 'all'], default='all')
    parser.add_argument('--no-planning', action='store_true', help='Disable world model planning')
    parser.add_argument('--planning-freq', type=float, default=0.3, help='Planning frequency (0-1)')
    parser.add_argument('--planning-horizon', type=int, default=5, help='Planning horizon (steps)')

    args = parser.parse_args()

    # Load agent
    print(f"Loading model: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)

    agent = ContextAwareDQN(obs_dim=95, action_dim=4)
    agent.load_state_dict(checkpoint['policy_net'])
    agent.eval()

    print(f"  Episodes trained: {len(checkpoint.get('episode_rewards', []))}")
    print(f"  Steps: {checkpoint.get('steps_done', 0)}")
    print(f"  Planning actions: {checkpoint.get('planning_count', 0)}")
    print(f"  Reactive actions: {checkpoint.get('reactive_count', 0)}")

    if 'context_episode_counts' in checkpoint:
        print(f"  Training context distribution:")
        for ctx, count in checkpoint['context_episode_counts'].items():
            print(f"    {ctx}: {count} episodes")

    # NEW: Display level progression info if available
    if 'context_levels' in checkpoint:
        print(f"  Level Progression:")
        for ctx in ['snake', 'balanced', 'survival']:
            level = checkpoint['context_levels'].get(ctx, 1)
            completions = checkpoint.get('level_completions', {}).get(ctx, 0)
            print(f"    {ctx:8s}: Level {level} - {completions} completions")

    # Load world model for planning
    world_model = None
    if not args.no_planning:
        base_path = args.model_path.replace('_policy.pth', '')
        world_model_path = f"{base_path}_world_model.pth"

        if os.path.exists(world_model_path):
            print(f"\nLoading world model for planning: {world_model_path}")
            world_model = WorldModelNetwork(state_dim=95, action_dim=4)
            wm_checkpoint = torch.load(world_model_path, map_location='cpu', weights_only=False)
            world_model.load_state_dict(wm_checkpoint['model'])
            world_model.eval()
            print(f"  Planning ENABLED: {args.planning_freq*100:.0f}% frequency, horizon {args.planning_horizon}")
        else:
            print(f"\n  Warning: World model not found at {world_model_path}")
            print(f"  Planning DISABLED - will use policy only")

    print()

    # Extract level configurations from checkpoint (if available)
    level_configs = {}
    if 'context_levels' in checkpoint:
        print("TESTING WITH TRAINING-LEVEL DIFFICULTY:")
        for ctx in ['snake', 'balanced', 'survival']:
            level = checkpoint['context_levels'].get(ctx, 1)
            completions = checkpoint.get('level_completions', {}).get(ctx, 0)
            print(f"  {ctx:8s}: Testing at Level {level} difficulty (trained {completions} completions)")

            # Map context to game difficulty
            if ctx == 'snake':
                level_configs['snake'] = {'level': level, 'enemies': 0}  # Snake = no enemies
            elif ctx == 'balanced':
                level_configs['pacman'] = {'level': level, 'enemies': level + 1}  # Pac-Man = balanced
            elif ctx == 'survival':
                level_configs['dungeon'] = {'level': level, 'enemies': level + 3}  # Dungeon = survival
        print()

    observer = TemporalFlowObserver()

    # Test games
    results = {}

    if args.game in ['snake', 'all']:
        game = SnakeGame(size=20)  # MATCH TRAINING: 20x20
        if 'snake' in level_configs:
            print(f"[Snake Game: Level {level_configs['snake']['level']} difficulty]\n")
        results['snake'] = test_game(agent, observer, game, "Snake", args.episodes,
                                     world_model=world_model,
                                     planning_freq=args.planning_freq,
                                     planning_horizon=args.planning_horizon)

    if args.game in ['pacman', 'all']:
        game = PacManGame(size=20)  # MATCH TRAINING: 20x20
        if 'pacman' in level_configs:
            print(f"[Pac-Man Game: Level {level_configs['pacman']['level']} difficulty]\n")
        results['pacman'] = test_game(agent, observer, game, "Pac-Man", args.episodes,
                                      world_model=world_model,
                                      planning_freq=args.planning_freq,
                                      planning_horizon=args.planning_horizon)

    if args.game in ['dungeon', 'all']:
        game = DungeonGame(size=20)  # MATCH TRAINING: 20x20
        if 'dungeon' in level_configs:
            print(f"[Dungeon Game: Level {level_configs['dungeon']['level']} difficulty]\n")
        results['dungeon'] = test_game(agent, observer, game, "Dungeon", args.episodes,
                                       world_model=world_model,
                                       planning_freq=args.planning_freq,
                                       planning_horizon=args.planning_horizon)

    # Summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY")
        print(f"{'='*60}")
        for game_name, result in results.items():
            print(f"{game_name:8s}: {result['avg_score']:6.2f} avg score")
        print()

        # Expected behavior check
        print("CONTEXT ADAPTATION CHECK:")
        snake_result = results.get('snake')
        if snake_result:
            snake_ctx = snake_result['context_counts']['snake']
            snake_total = sum(snake_result['context_counts'].values())
            snake_pct = (snake_ctx / snake_total * 100) if snake_total > 0 else 0
            print(f"  Snake game: {snake_pct:.1f}% 'snake' context detected")
            if snake_pct > 80:
                print("    [GOOD] Agent correctly detects no-entity context")
            else:
                print(f"    [WARNING] Should be >80% for pure collection task")


if __name__ == '__main__':
    main()
