"""
Simple Pac-Man Demo - Test Best Model Performance

Runs the best trained model on Pac-Man game and shows real performance.
Uses Episode 260 checkpoint (best performance: 3152.94 avg reward)

Usage:
    python demo_pacman.py
    python demo_pacman.py --episodes 20 --speed 0.1
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import time
import argparse
from core.planning_test_games import PacManGame
from context_aware_agent import ContextAwareDQN, infer_context_from_observation, add_context_to_observation
from core.temporal_observer import TemporalFlowObserver
from core.world_model import WorldModelNetwork
from train_context_aware_advanced import ContinuousMotivationRewardSystem


def visualize_game_state(game, step, score):
    """Simple ASCII visualization of Pac-Man game"""
    grid = [[' ' for _ in range(game.size)] for _ in range(game.size)]

    # Draw walls
    for x, y in game.walls:
        grid[y][x] = '#'

    # Draw pellets
    for x, y in game.pellets:
        grid[y][x] = '.'

    # Draw ghosts
    for ghost in game.ghosts:
        x, y = ghost['pos']
        grid[y][x] = 'G'

    # Draw Pac-Man (overwrites if on pellet)
    px, py = game.pacman_pos
    grid[py][px] = 'P'

    # Clear screen (works on most terminals)
    print('\033[2J\033[H', end='')

    # Print game state
    print("="*60)
    print(f"PAC-MAN DEMO - Best Model (Episode 260)")
    print("="*60)
    print(f"Step: {step:4d} | Score: {score:3d} | Pellets: {len(game.pellets):3d} | Lives: {game.lives}")
    print("-"*60)

    # Print grid
    for row in grid:
        print(''.join(row))

    print("-"*60)
    print("Legend: P=Pac-Man, G=Ghost, .=Pellet, #=Wall")


def plan_action(agent, world_model, state, planning_horizon=5):
    """Use world model to plan best action via lookahead"""
    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    best_action = None
    best_return = -float('inf')

    # Try each action
    for action in range(4):
        total_return = 0.0
        num_rollouts = 5
        for _ in range(num_rollouts):
            rollout_return = simulate_rollout(agent, world_model, state_tensor, action, planning_horizon)
            total_return += rollout_return
        avg_return = total_return / num_rollouts
        if avg_return > best_return:
            best_return = avg_return
            best_action = action

    return best_action


def simulate_rollout(agent, world_model, state, first_action, planning_horizon=5):
    """Simulate one trajectory using world model"""
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


def calculate_distances(game_state):
    """Calculate nearest pellet and enemy distances"""
    agent_pos = game_state['agent_pos']

    # Nearest pellet
    nearest_pellet_dist = None
    if game_state['rewards']:
        distances = [abs(p[0] - agent_pos[0]) + abs(p[1] - agent_pos[1])
                    for p in game_state['rewards']]
        nearest_pellet_dist = min(distances) if distances else None

    # Nearest enemy
    nearest_enemy_dist = None
    if game_state['entities']:
        distances = [abs(e['pos'][0] - agent_pos[0]) + abs(e['pos'][1] - agent_pos[1])
                    for e in game_state['entities']]
        nearest_enemy_dist = min(distances) if distances else None

    return nearest_pellet_dist, nearest_enemy_dist


def run_episode(agent, observer, game, world_model=None, planning_freq=0.3,
                visualize=False, speed=0.0, use_motivation=False):
    """Run one episode of Pac-Man"""
    game.reset()
    observer.reset()
    game_state = game._get_game_state()

    # Initialize continuous motivation system if requested
    motivation_system = None
    if use_motivation:
        motivation_system = ContinuousMotivationRewardSystem(context_name='balanced')
        motivation_system.reset()

    total_reward = 0
    total_base_reward = 0
    total_bonus_reward = 0
    steps = 0
    done = False
    pellets_collected = 0
    initial_pellets = len(game.pellets)

    planning_actions = 0
    reactive_actions = 0

    reward_breakdown_totals = {
        'env': 0.0,
        'approach': 0.0,
        'combo': 0.0,
        'risk': 0.0,
        'streak': 0.0,
        'level': 0.0,
        'death_penalty': 0.0
    }

    while not done and steps < 1000:
        # Visualize
        if visualize:
            visualize_game_state(game, steps, game_state['score'])
            if speed > 0:
                time.sleep(speed)

        # Get observation
        obs = observer.observe(game_state)
        context_vector = infer_context_from_observation(obs)
        obs_with_context = add_context_to_observation(obs, context_vector)

        # Get action (with planning if available)
        if world_model is not None and np.random.random() < planning_freq:
            action = plan_action(agent, world_model, obs_with_context, planning_horizon=5)
            planning_actions += 1
        else:
            action = agent.get_action(obs_with_context, epsilon=0.0)
            reactive_actions += 1

        # Execute action
        prev_pellets = len(game.pellets)
        prev_lives = game.lives
        game_state, base_reward, done = game.step(action)
        steps += 1

        # Track pellets collected and deaths
        collected = len(game.pellets) < prev_pellets
        died = game.lives < prev_lives

        if collected:
            pellets_collected += 1

        # Apply continuous motivation system if enabled
        if motivation_system:
            # Calculate distances for motivation system
            nearest_pellet_dist, nearest_enemy_dist = calculate_distances(game_state)

            # Build info dict
            info = {
                'collected_reward': collected,
                'died': died
            }

            # Calculate enhanced reward
            enhanced_reward, breakdown = motivation_system.calculate_reward(
                base_reward, info, nearest_pellet_dist, nearest_enemy_dist
            )

            total_reward += enhanced_reward
            total_base_reward += base_reward
            total_bonus_reward += (enhanced_reward - base_reward)

            # Track breakdown
            for key, value in breakdown.items():
                reward_breakdown_totals[key] += value
        else:
            # Use base reward only
            total_reward += base_reward
            total_base_reward += base_reward

    if visualize:
        visualize_game_state(game, steps, game_state['score'])
        print(f"\nEpisode finished!")
        print(f"  Total Reward: {total_reward:.2f}")
        if use_motivation:
            print(f"    Base Reward: {total_base_reward:.2f}")
            print(f"    Bonus Reward: {total_bonus_reward:.2f}")
            print(f"\n  Reward Breakdown:")
            for key, value in reward_breakdown_totals.items():
                if abs(value) > 0.1:
                    print(f"    {key:15s}: {value:+8.2f}")
        print(f"  Pellets Collected: {pellets_collected}/{initial_pellets}")
        print(f"  Planning Actions: {planning_actions}")
        print(f"  Reactive Actions: {reactive_actions}")
        print(f"  Survival: {steps} steps")
        input("\nPress Enter to continue...")

    result = {
        'score': game_state['score'],
        'reward': total_reward,
        'base_reward': total_base_reward,
        'bonus_reward': total_bonus_reward,
        'steps': steps,
        'pellets_collected': pellets_collected,
        'initial_pellets': initial_pellets,
        'completion': pellets_collected / initial_pellets if initial_pellets > 0 else 0,
        'planning_actions': planning_actions,
        'reactive_actions': reactive_actions
    }

    if use_motivation:
        result['reward_breakdown'] = reward_breakdown_totals
        stats = motivation_system.get_stats()
        result.update(stats)

    return result


def main():
    parser = argparse.ArgumentParser(description='Pac-Man Demo with Best Model')
    parser.add_argument('--model', type=str,
                       default='checkpoints/context_aware_advanced_20251118_202829_best_policy.pth',
                       help='Path to model checkpoint (default: best Episode 260)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to run (default: 10)')
    parser.add_argument('--visualize', action='store_true',
                       help='Show ASCII visualization of game')
    parser.add_argument('--speed', type=float, default=0.05,
                       help='Visualization speed in seconds (default: 0.05)')
    parser.add_argument('--planning-freq', type=float, default=0.3,
                       help='Planning frequency 0-1 (default: 0.3)')
    parser.add_argument('--use-motivation', action='store_true',
                       help='Use Continuous Motivation System (training rewards)')

    args = parser.parse_args()

    print("="*60)
    print("PAC-MAN PERFORMANCE DEMO")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Planning Frequency: {args.planning_freq*100:.0f}%")
    print(f"Reward System: {'CONTINUOUS MOTIVATION (Training)' if args.use_motivation else 'BASE GAME ONLY (Simple)'}")
    print("="*60)

    # Load agent
    try:
        checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
        agent = ContextAwareDQN(obs_dim=95, action_dim=4)
        agent.load_state_dict(checkpoint['policy_net'])
        agent.eval()
        print(f"[OK] Loaded agent from Episode {len(checkpoint.get('episode_rewards', []))}")
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return

    # Load world model
    world_model = None
    world_model_path = args.model.replace('_policy.pth', '_world_model.pth')
    if os.path.exists(world_model_path):
        try:
            world_model = WorldModelNetwork(state_dim=95, action_dim=4)
            wm_checkpoint = torch.load(world_model_path, map_location='cpu', weights_only=False)
            world_model.load_state_dict(wm_checkpoint['model'])
            world_model.eval()
            print(f"[OK] Loaded world model for planning")
        except Exception as e:
            print(f"[WARN] World model not available: {e}")
    else:
        print(f"[WARN] World model not found (running without planning)")

    print()

    # Run episodes
    game = PacManGame(size=20)
    observer = TemporalFlowObserver()

    results = []
    for episode in range(args.episodes):
        visualize = args.visualize if episode == 0 else False  # Only visualize first episode
        result = run_episode(agent, observer, game, world_model,
                           planning_freq=args.planning_freq,
                           visualize=visualize,
                           speed=args.speed,
                           use_motivation=args.use_motivation)
        results.append(result)

        if not visualize:
            reward_str = f"Reward={result['reward']:7.1f}"
            if args.use_motivation:
                reward_str += f" (Base={result['base_reward']:6.1f} +Bonus={result['bonus_reward']:6.1f})"
            print(f"Episode {episode+1:3d}: Score={result['score']:3d} "
                  f"Pellets={result['pellets_collected']:3d}/{result['initial_pellets']:3d} "
                  f"({result['completion']*100:5.1f}%) "
                  f"{reward_str} "
                  f"Steps={result['steps']:4d}")

    # Print summary
    print()
    print("="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    scores = [r['score'] for r in results]
    rewards = [r['reward'] for r in results]
    completions = [r['completion'] for r in results]
    steps = [r['steps'] for r in results]

    print(f"Average Score:        {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Average Reward:       {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Average Completion:   {np.mean(completions)*100:.1f}% ± {np.std(completions)*100:.1f}%")
    print(f"Average Steps:        {np.mean(steps):.1f} ± {np.std(steps):.1f}")
    print()
    print(f"Best Score:           {max(scores)}")
    print(f"Best Completion:      {max(completions)*100:.1f}%")
    print(f"Best Reward:          {max(rewards):.1f}")
    print()

    total_planning = sum(r['planning_actions'] for r in results)
    total_reactive = sum(r['reactive_actions'] for r in results)
    total_actions = total_planning + total_reactive

    print(f"Planning Actions:     {total_planning} ({total_planning/total_actions*100:.1f}%)")
    print(f"Reactive Actions:     {total_reactive} ({total_reactive/total_actions*100:.1f}%)")
    print()

    # Motivation breakdown if applicable
    if args.use_motivation and 'reward_breakdown' in results[0]:
        print("MOTIVATION REWARD BREAKDOWN (Average per episode):")
        print("-"*60)
        breakdown_sums = {
            'env': 0, 'approach': 0, 'combo': 0,
            'risk': 0, 'streak': 0, 'level': 0, 'death_penalty': 0
        }
        for r in results:
            if 'reward_breakdown' in r:
                for key, value in r['reward_breakdown'].items():
                    breakdown_sums[key] += value

        for key, value in breakdown_sums.items():
            avg_value = value / len(results)
            if abs(avg_value) > 0.1:
                print(f"  {key:15s}: {avg_value:+8.2f}")
        print()

        # Show base vs bonus split
        base_rewards = [r['base_reward'] for r in results]
        bonus_rewards = [r['bonus_reward'] for r in results]
        print(f"Base Reward Avg:      {np.mean(base_rewards):8.2f} (game mechanics)")
        print(f"Bonus Reward Avg:     {np.mean(bonus_rewards):+8.2f} (motivation bonuses)")
        print(f"Total Reward Avg:     {np.mean(rewards):8.2f}")
        print()

    # Performance assessment
    avg_completion = np.mean(completions)
    if avg_completion > 0.8:
        grade = "EXCELLENT ***"
    elif avg_completion > 0.6:
        grade = "GOOD **"
    elif avg_completion > 0.4:
        grade = "FAIR *"
    else:
        grade = "NEEDS IMPROVEMENT"

    print(f"Overall Performance:  {grade}")
    print("="*60)


if __name__ == '__main__':
    main()
