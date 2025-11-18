"""
Compare Snake game behavior between test and visual environments
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from core.planning_test_games import SnakeGame
from context_aware_agent import ContextAwareDQN, infer_context_from_observation, add_context_to_observation
from core.temporal_observer import TemporalFlowObserver
from core.world_model import WorldModelNetwork

def _plan_action(agent, world_model, state, planning_horizon=5):
    """Use world model to plan best action via lookahead"""
    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    best_action = None
    best_return = -float('inf')

    # Try each action
    for action in range(4):
        total_return = 0.0
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


def run_test_environment(agent, observer, world_model, planning_freq=0.3):
    """Run one episode in TEST environment style"""
    print("\n" + "="*60)
    print("TEST ENVIRONMENT (test_context_aware.py style)")
    print("="*60)

    game = SnakeGame(size=20)
    game.reset()
    observer.reset()
    game_state = game._get_game_state()

    total_reward = 0
    steps = 0
    done = False
    food_collected = 0

    print(f"Initial state:")
    print(f"  Agent pos: {game_state['agent_pos']}")
    print(f"  Food pos: {game_state['rewards'][0]}")
    print(f"  Grid size: {game_state.get('grid_size', 'MISSING!')}")

    while not done and steps < 200:  # Limit to 200 steps for comparison
        obs = observer.observe(game_state)
        context_vector = infer_context_from_observation(obs)
        obs_with_context = add_context_to_observation(obs, context_vector)

        # Get action with planning
        if world_model is not None and np.random.random() < planning_freq:
            action = _plan_action(agent, world_model, obs_with_context, planning_horizon=5)
        else:
            action = agent.get_action(obs_with_context, epsilon=0.0)

        # Execute action
        game_state, reward, done = game.step(action)
        total_reward += reward
        steps += 1

        # Check for food collection
        if reward > 10:  # Food collected
            food_collected += 1
            print(f"  Step {steps}: Food collected! Score: {game_state['score']}, Reward: {reward:.1f}")

    print(f"\nResults:")
    print(f"  Final score: {game_state['score']}")
    print(f"  Food collected: {food_collected}")
    print(f"  Total reward: {total_reward:.1f}")
    print(f"  Steps: {steps}")
    print(f"  Done: {done}")

    return game_state['score'], steps, total_reward


def run_visual_environment(agent, observer, world_model, planning_freq=0.3):
    """Run one episode in VISUAL environment style"""
    print("\n" + "="*60)
    print("VISUAL ENVIRONMENT (context_aware_visual_games.py style)")
    print("="*60)

    game = SnakeGame(size=20)
    game_state = game.reset()  # Store returned state (like visual does)
    observer.reset()

    total_reward = 0
    steps = 0
    food_collected = 0

    print(f"Initial state:")
    print(f"  Agent pos: {game_state['agent_pos']}")
    print(f"  Food pos: {game_state['rewards'][0]}")
    print(f"  Grid size: {game_state.get('grid_size', 'MISSING!')}")

    while not game_state['done'] and steps < 200:  # Limit to 200 steps for comparison
        # Get observation
        obs = observer.observe(game_state)

        # Infer context
        context_vector = infer_context_from_observation(obs)

        # Add context to observation
        obs_with_context = add_context_to_observation(obs, context_vector)

        # Get action with planning
        if world_model is not None and np.random.random() < planning_freq:
            action = _plan_action(agent, world_model, obs_with_context, planning_horizon=5)
        else:
            action = agent.get_action(obs_with_context, epsilon=0.0)

        # Take step (like visual does)
        prev_score = game_state['score']
        game_state, reward, done = game.step(action)
        total_reward += reward
        steps += 1

        # Check for food collection
        if game_state['score'] > prev_score:
            food_collected += 1
            print(f"  Step {steps}: Food collected! Score: {game_state['score']}, Reward: {reward:.1f}")

    print(f"\nResults:")
    print(f"  Final score: {game_state['score']}")
    print(f"  Food collected: {food_collected}")
    print(f"  Total reward: {total_reward:.1f}")
    print(f"  Steps: {steps}")
    print(f"  Done: {game_state['done']}")

    return game_state['score'], steps, total_reward


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compare Snake environments')
    parser.add_argument('model_path', help='Path to policy checkpoint')
    args = parser.parse_args()

    # Load agent
    print(f"Loading model: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    agent = ContextAwareDQN(obs_dim=95, action_dim=4)
    agent.load_state_dict(checkpoint['policy_net'])
    agent.eval()

    # Load world model
    base_path = args.model_path.replace('_policy.pth', '')
    world_model_path = f"{base_path}_world_model.pth"
    world_model = None
    if os.path.exists(world_model_path):
        print(f"Loading world model: {world_model_path}")
        world_model = WorldModelNetwork(state_dim=95, action_dim=4)
        wm_checkpoint = torch.load(world_model_path, map_location='cpu', weights_only=False)
        world_model.load_state_dict(wm_checkpoint['model'])
        world_model.eval()

    # Run 5 episodes in each environment
    print("\n" + "="*60)
    print("COMPARING ENVIRONMENTS (5 episodes each)")
    print("="*60)

    test_scores = []
    visual_scores = []

    for i in range(5):
        print(f"\n{'='*60}")
        print(f"EPISODE {i+1}")
        print(f"{'='*60}")

        observer_test = TemporalFlowObserver()
        score, steps, reward = run_test_environment(agent, observer_test, world_model)
        test_scores.append(score)

        observer_visual = TemporalFlowObserver()
        score, steps, reward = run_visual_environment(agent, observer_visual, world_model)
        visual_scores.append(score)

    # Compare results
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Test environment avg score: {np.mean(test_scores):.2f} ± {np.std(test_scores):.2f}")
    print(f"  Scores: {test_scores}")
    print()
    print(f"Visual environment avg score: {np.mean(visual_scores):.2f} ± {np.std(visual_scores):.2f}")
    print(f"  Scores: {visual_scores}")
    print()

    if np.mean(test_scores) > np.mean(visual_scores):
        diff = np.mean(test_scores) - np.mean(visual_scores)
        print(f"TEST environment performs BETTER by {diff:.2f} avg score")
    elif np.mean(visual_scores) > np.mean(test_scores):
        diff = np.mean(visual_scores) - np.mean(test_scores)
        print(f"VISUAL environment performs BETTER by {diff:.2f} avg score")
    else:
        print("Both environments perform IDENTICALLY")


if __name__ == '__main__':
    main()
