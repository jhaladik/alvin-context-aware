"""
Train Temporal Enhanced Agent - Quick fix for ghost prediction

Fine-tunes existing model with temporal buffer enhancement
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from datetime import datetime
from collections import deque

from context_aware_agent import ContextAwareDQN, infer_context_from_observation, add_context_to_observation
from core.temporal_observer import TemporalFlowObserver
from core.temporal_buffer_enhancement import TemporalBufferEnhancement, GhostEnsemblePredictor
from core.planning_test_games import PacManGame, SnakeGame, DungeonGame
from train_context_aware_advanced import PrioritizedReplayBuffer


class TemporalEnhancedAgent(nn.Module):
    """
    Enhanced agent with temporal buffering

    Architecture:
    - Base: Existing trained ContextAwareDQN (mostly frozen)
    - Enhancement: TemporalBufferEnhancement (trainable)
    - Predictor: GhostEnsemblePredictor (no params, heuristic)
    """

    def __init__(self, base_agent, freeze_base=True):
        super().__init__()

        self.base_agent = base_agent
        self.temporal_enhancement = TemporalBufferEnhancement(obs_dim=95, buffer_size=50)
        self.ghost_predictor = GhostEnsemblePredictor()

        # Freeze base agent during fine-tuning
        if freeze_base:
            for param in self.base_agent.parameters():
                param.requires_grad = False

        # Only train enhancement layers
        self.trainable_params = list(self.temporal_enhancement.parameters())

    def get_q_values(self, obs):
        """Get Q-values with temporal enhancement

        Args:
            obs: (obs_dim,) for single or (batch, obs_dim) for batch

        Returns:
            q_values: (1, action_dim) for single or (batch, action_dim) for batch
            uncertainty: scalar tensor
        """
        # Enhance observation (preserves input shape)
        enhanced_obs, uncertainty = self.temporal_enhancement.enhance_observation(obs)

        # Get Q-values from base agent
        # enhanced_obs is (obs_dim,) for single or (batch, obs_dim) for batch
        is_batch = len(enhanced_obs.shape) == 2

        if is_batch:
            # Already has batch dimension
            q_values = self.base_agent.get_combined_q(enhanced_obs)
        else:
            # Need to add batch dimension
            q_values = self.base_agent.get_combined_q(enhanced_obs.unsqueeze(0))

        return q_values, uncertainty

    def get_action(self, obs, ghost_positions=None, agent_pos=None, reward_pos=None,
                   epsilon=0.0, use_ensemble=True):
        """
        Get action with optional ensemble prediction
        """
        # Update temporal buffer
        obs_tensor = torch.FloatTensor(obs) if not isinstance(obs, torch.Tensor) else obs
        self.temporal_enhancement.update_buffers(obs_tensor)

        # Get Q-values and uncertainty
        q_values, uncertainty = self.get_q_values(obs_tensor)

        # High uncertainty + ghosts present → use ensemble
        if (use_ensemble and ghost_positions is not None and
            agent_pos is not None and reward_pos is not None and
            uncertainty > 0.6):

            # Update ghost predictor
            self.ghost_predictor.update(ghost_positions)

            # Get ensemble predictions
            predictions = self.ghost_predictor.predict_ensemble(agent_pos, steps_ahead=5)

            if predictions and predictions['expected']:
                # Find safe zones
                safe_zones = self.ghost_predictor.compute_safe_zones(agent_pos, predictions)

                if safe_zones:
                    # Use ensemble-based action
                    action = self.ghost_predictor.get_best_action(
                        agent_pos, safe_zones, reward_pos
                    )
                    return action, True  # True = used ensemble

        # Normal Q-value based action
        if np.random.random() < epsilon:
            action = np.random.randint(4)
        else:
            action = q_values.argmax().item()

        return action, False  # False = used Q-values

    def reset(self):
        """Reset temporal buffers for new episode"""
        self.temporal_enhancement.reset()


def run_episode(agent, observer, game, optimizer=None, buffer=None,
                train_mode=True, use_ensemble=True, gamma=0.99):
    """
    Run one episode with temporal enhancement

    Returns episode stats
    """
    game.reset()
    observer.reset()
    agent.reset()

    game_state = game._get_game_state()
    total_reward = 0
    steps = 0
    done = False

    ensemble_uses = 0
    total_loss = 0
    updates = 0

    while not done and steps < 1000:
        # Get observation
        obs = observer.observe(game_state)
        context_vector = infer_context_from_observation(obs)
        obs_with_context = add_context_to_observation(obs, context_vector)

        # Extract ghost positions and rewards for ensemble
        ghost_positions = [e['pos'] for e in game_state.get('entities', [])]
        agent_pos = game_state['agent_pos']
        rewards = game_state.get('rewards', [])
        reward_pos = rewards[0] if rewards else None

        # Get action
        epsilon = 0.1 if train_mode else 0.0
        action, used_ensemble = agent.get_action(
            obs_with_context,
            ghost_positions=ghost_positions,
            agent_pos=agent_pos,
            reward_pos=reward_pos,
            epsilon=epsilon,
            use_ensemble=use_ensemble
        )

        if used_ensemble:
            ensemble_uses += 1

        # Execute action
        next_game_state, reward, done = game.step(action)
        steps += 1
        total_reward += reward

        # Store transition if training
        if train_mode and buffer is not None:
            next_obs = observer.observe(next_game_state)
            next_context = infer_context_from_observation(next_obs)
            next_obs_with_context = add_context_to_observation(next_obs, next_context)

            buffer.add((obs_with_context, action, reward, next_obs_with_context, done))

        # Train if buffer has enough samples
        if train_mode and buffer is not None and len(buffer) > 128:
            # Sample batch
            transitions, indices, weights = buffer.sample(32)

            if transitions is None:
                continue

            # Unpack transitions (use numpy stacking for performance)
            states = torch.FloatTensor(np.array([t[0] for t in transitions]))
            actions = torch.LongTensor(np.array([t[1] for t in transitions]))
            rewards_batch = torch.FloatTensor(np.array([t[2] for t in transitions]))
            next_states = torch.FloatTensor(np.array([t[3] for t in transitions]))
            dones = torch.FloatTensor(np.array([t[4] for t in transitions]))

            # Compute current Q values
            current_q, _ = agent.get_q_values(states)
            current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Compute target Q values
            with torch.no_grad():
                next_q, _ = agent.get_q_values(next_states)
                next_q_max = next_q.max(1)[0]
                target_q = rewards_batch + gamma * next_q_max * (1 - dones)

            # Compute loss
            loss = nn.functional.mse_loss(current_q, target_q)

            # Optimize (only enhancement parameters)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.trainable_params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            updates += 1

        game_state = next_game_state

    return {
        'reward': total_reward,
        'steps': steps,
        'score': game_state['score'],
        'ensemble_uses': ensemble_uses,
        'ensemble_pct': ensemble_uses / steps * 100 if steps > 0 else 0,
        'loss': total_loss / max(updates, 1)
    }


def main():
    parser = argparse.ArgumentParser(description='Train Temporal Enhanced Agent')
    parser.add_argument('--base-model', type=str, required=True,
                       help='Path to base model checkpoint')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of fine-tuning episodes')
    parser.add_argument('--use-ensemble', action='store_true',
                       help='Use ghost ensemble prediction')
    parser.add_argument('--freeze-base', action='store_true', default=True,
                       help='Freeze base agent weights')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate for enhancement layers')
    parser.add_argument('--test-freq', type=int, default=10,
                       help='Test every N episodes')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')

    args = parser.parse_args()

    print("="*70)
    print("TEMPORAL ENHANCED AGENT - FINE-TUNING")
    print("="*70)
    print(f"Base model: {args.base_model}")
    print(f"Fine-tune episodes: {args.episodes}")
    print(f"Ensemble prediction: {args.use_ensemble}")
    print(f"Freeze base: {args.freeze_base}")
    print(f"Learning rate: {args.lr}")
    print("="*70)
    print()

    # Load base model
    print("Loading base model...")
    checkpoint = torch.load(args.base_model, map_location='cpu', weights_only=False)
    base_agent = ContextAwareDQN(obs_dim=95, action_dim=4)
    base_agent.load_state_dict(checkpoint['policy_net'])
    base_agent.eval()
    print(f"  Loaded from episode {len(checkpoint.get('episode_rewards', []))}")
    print()

    # Create enhanced agent
    print("Creating temporal enhanced agent...")
    agent = TemporalEnhancedAgent(base_agent, freeze_base=args.freeze_base)

    # Count parameters
    total_params = sum(p.numel() for p in agent.parameters())
    trainable_params = sum(p.numel() for p in agent.trainable_params)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable (enhancement only): {trainable_params:,}")
    print(f"  Frozen (base agent): {total_params - trainable_params:,}")
    print()

    # Setup optimizer (only for enhancement layers)
    optimizer = optim.Adam(agent.trainable_params, lr=args.lr)

    # Replay buffer
    buffer = PrioritizedReplayBuffer(capacity=10000)

    # Create games
    games = {
        'pacman': PacManGame(size=20),
        'snake': SnakeGame(size=20),
        'dungeon': DungeonGame(size=20)
    }

    observer = TemporalFlowObserver()

    # Training loop
    print("Starting fine-tuning...")
    print()

    best_pacman_score = 0
    episode_rewards = []

    for episode in range(args.episodes):
        # Alternate games (focus on Pac-Man)
        if episode % 3 == 0:
            game_name = 'pacman'
        elif episode % 3 == 1:
            game_name = 'snake'
        else:
            game_name = 'dungeon'

        game = games[game_name]

        # Train episode
        stats = run_episode(
            agent, observer, game,
            optimizer=optimizer,
            buffer=buffer,
            train_mode=True,
            use_ensemble=args.use_ensemble,
            gamma=args.gamma
        )

        episode_rewards.append(stats['reward'])

        print(f"Episode {episode+1:3d}/{args.episodes} ({game_name:7s}): "
              f"Score={stats['score']:3d} Reward={stats['reward']:6.1f} "
              f"Steps={stats['steps']:3d} Loss={stats['loss']:.4f} "
              f"Ensemble={stats['ensemble_pct']:4.1f}%")

        # Test on Pac-Man periodically
        if (episode + 1) % args.test_freq == 0:
            print()
            print(f"--- Testing after episode {episode+1} ---")

            test_rewards = []
            test_scores = []

            for test_ep in range(10):
                test_stats = run_episode(
                    agent, observer, games['pacman'],
                    optimizer=None,
                    buffer=None,
                    train_mode=False,
                    use_ensemble=args.use_ensemble
                )
                test_rewards.append(test_stats['reward'])
                test_scores.append(test_stats['score'])

            avg_score = np.mean(test_scores)
            avg_reward = np.mean(test_rewards)

            print(f"Pac-Man test (10 episodes):")
            print(f"  Avg Score: {avg_score:.2f} ± {np.std(test_scores):.2f}")
            print(f"  Avg Reward: {avg_reward:.2f} ± {np.std(test_rewards):.2f}")
            print(f"  Best Score: {max(test_scores)}")

            if avg_score > best_pacman_score:
                best_pacman_score = avg_score
                # Save checkpoint
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"checkpoints/temporal_enhanced_{timestamp}_best.pth"

                torch.save({
                    'episode': episode + 1,
                    'base_model_path': args.base_model,
                    'enhancement_state_dict': agent.temporal_enhancement.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_pacman_score': best_pacman_score,
                    'episode_rewards': episode_rewards,
                    'config': vars(args)
                }, save_path)

                print(f"  ✓ New best! Saved to {save_path}")

            print()

    # Final test
    print()
    print("="*70)
    print("FINAL EVALUATION")
    print("="*70)

    for game_name, game in games.items():
        print(f"\n{game_name.upper()} (50 episodes):")

        scores = []
        rewards = []
        ensemble_uses = []

        for _ in range(50):
            stats = run_episode(
                agent, observer, game,
                train_mode=False,
                use_ensemble=args.use_ensemble
            )
            scores.append(stats['score'])
            rewards.append(stats['reward'])
            ensemble_uses.append(stats['ensemble_pct'])

        print(f"  Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
        print(f"  Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"  Best Score: {max(scores)}")
        print(f"  Ensemble Usage: {np.mean(ensemble_uses):.1f}%")

    print()
    print("Fine-tuning complete!")
    print(f"Best Pac-Man score: {best_pacman_score:.2f}")


if __name__ == '__main__':
    main()
