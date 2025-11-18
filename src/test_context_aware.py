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
from temporal_observer import TemporalFlowObserver
from planning_test_games import SnakeGame, PacManGame, DungeonGame


def test_game(agent, observer, game, game_name, num_episodes=50):
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

            # Get action
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

    args = parser.parse_args()

    # Load agent
    print(f"Loading model: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu')

    agent = ContextAwareDQN(obs_dim=95, action_dim=4)
    agent.load_state_dict(checkpoint['policy_net'])
    agent.eval()

    print(f"  Episodes trained: {len(checkpoint.get('episode_rewards', []))}")
    print(f"  Steps: {checkpoint.get('steps_done', 0)}")

    if 'context_episode_counts' in checkpoint:
        print(f"  Training context distribution:")
        for ctx, count in checkpoint['context_episode_counts'].items():
            print(f"    {ctx}: {count} episodes")

    observer = TemporalFlowObserver()

    # Test games
    results = {}

    if args.game in ['snake', 'all']:
        game = SnakeGame(size=15)
        results['snake'] = test_game(agent, observer, game, "Snake", args.episodes)

    if args.game in ['pacman', 'all']:
        game = PacManGame(size=15)
        results['pacman'] = test_game(agent, observer, game, "Pac-Man", args.episodes)

    if args.game in ['dungeon', 'all']:
        game = DungeonGame(size=20)
        results['dungeon'] = test_game(agent, observer, game, "Dungeon", args.episodes)

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
