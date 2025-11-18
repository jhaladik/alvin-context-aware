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


def run_episode(agent, observer, game, world_model=None, planning_freq=0.3, visualize=False, speed=0.0):
    """Run one episode of Pac-Man"""
    game.reset()
    observer.reset()
    game_state = game._get_game_state()

    total_reward = 0
    steps = 0
    done = False
    pellets_collected = 0
    initial_pellets = len(game.pellets)

    planning_actions = 0
    reactive_actions = 0

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
        game_state, reward, done = game.step(action)
        total_reward += reward
        steps += 1

        # Track pellets collected
        if len(game.pellets) < prev_pellets:
            pellets_collected += 1

    if visualize:
        visualize_game_state(game, steps, game_state['score'])
        print(f"\nEpisode finished!")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Pellets Collected: {pellets_collected}/{initial_pellets}")
        print(f"  Planning Actions: {planning_actions}")
        print(f"  Reactive Actions: {reactive_actions}")
        print(f"  Survival: {steps} steps")
        input("\nPress Enter to continue...")

    return {
        'score': game_state['score'],
        'reward': total_reward,
        'steps': steps,
        'pellets_collected': pellets_collected,
        'initial_pellets': initial_pellets,
        'completion': pellets_collected / initial_pellets if initial_pellets > 0 else 0,
        'planning_actions': planning_actions,
        'reactive_actions': reactive_actions
    }


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

    args = parser.parse_args()

    print("="*60)
    print("PAC-MAN PERFORMANCE DEMO")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Planning Frequency: {args.planning_freq*100:.0f}%")
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
                           speed=args.speed)
        results.append(result)

        if not visualize:
            print(f"Episode {episode+1:3d}: Score={result['score']:3d} "
                  f"Pellets={result['pellets_collected']:3d}/{result['initial_pellets']:3d} "
                  f"({result['completion']*100:5.1f}%) "
                  f"Reward={result['reward']:7.1f} "
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
