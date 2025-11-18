"""
Context-Aware Training with Mixed Scenarios

Solves the spurious correlation problem by training on multiple contexts:
- Snake mode (0 entities): Pure collection task
- Balanced mode (2-3 entities): Tactical gameplay
- Survival mode (4+ entities): High threat avoidance

The agent learns to adapt behavior based on context signal.

Usage:
    python train_context_aware.py --episodes 5000
    python train_context_aware.py --episodes 2000 --log-every 50
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))

import torch
import numpy as np
import argparse
import os
from datetime import datetime
from collections import deque
from context_aware_agent import (
    ContextAwareDQN,
    infer_context_from_observation,
    add_context_to_observation
)
from temporal_env import TemporalRandom2DEnv
from world_model import WorldModelNetwork, WorldModelTrainer


class ContextAwareTrainer:
    """
    Trainer that samples different contexts and trains context-aware agent.

    Context Distribution:
    - 30% Snake mode (0 entities)
    - 50% Balanced mode (2-3 entities)
    - 20% Survival mode (4+ entities)
    """

    def __init__(
        self,
        env_size=20,
        num_rewards=10,
        buffer_size=100000,
        batch_size=64,
        gamma=0.99,
        lr_policy=0.0001,
        lr_world_model=0.0003,
        target_update_freq=500
    ):
        self.env_size = env_size
        self.num_rewards = num_rewards
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq

        # Context distribution (probabilities sum to 1.0)
        self.context_distribution = {
            'snake': 0.30,      # 0 entities
            'balanced': 0.50,   # 2-3 entities
            'survival': 0.20    # 4+ entities
        }

        # Context to num_entities mapping
        self.context_configs = {
            'snake': (0, 0),      # (min, max) entities
            'balanced': (2, 3),
            'survival': (4, 6)
        }

        # Networks
        self.policy_net = ContextAwareDQN(obs_dim=95, action_dim=4)
        self.target_net = ContextAwareDQN(obs_dim=95, action_dim=4)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.world_model = WorldModelNetwork(state_dim=95, action_dim=4)

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=lr_world_model)

        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)

        # Training stats
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.world_model_losses = []
        self.steps_done = 0

        # Context tracking
        self.context_episode_counts = {'snake': 0, 'balanced': 0, 'survival': 0}
        self.context_avg_rewards = {'snake': [], 'balanced': [], 'survival': []}

        print("=" * 60)
        print("CONTEXT-AWARE TRAINER INITIALIZED")
        print("=" * 60)
        print(f"Policy network: {sum(p.numel() for p in self.policy_net.parameters()):,} parameters")
        print(f"World model: {sum(p.numel() for p in self.world_model.parameters()):,} parameters")
        print(f"Input dim: 95 (92 temporal + 3 context)")
        print()
        print("CONTEXT DISTRIBUTION:")
        for context, prob in self.context_distribution.items():
            entities = self.context_configs[context]
            print(f"  {context:8s}: {prob*100:4.0f}% (entities: {entities[0]}-{entities[1]})")
        print()

    def sample_context(self):
        """Sample a context based on distribution"""
        contexts = list(self.context_distribution.keys())
        probs = list(self.context_distribution.values())
        return np.random.choice(contexts, p=probs)

    def create_env_for_context(self, context):
        """Create environment with appropriate num_entities for context"""
        min_entities, max_entities = self.context_configs[context]
        num_entities = np.random.randint(min_entities, max_entities + 1)

        env = TemporalRandom2DEnv(
            grid_size=(self.env_size, self.env_size),
            num_entities=num_entities,
            num_rewards=self.num_rewards
        )
        return env, num_entities

    def get_context_vector(self, context):
        """Get one-hot context vector for training"""
        if context == 'snake':
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif context == 'balanced':
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:  # survival
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def train_episode(self, context, epsilon):
        """Train one episode with given context"""
        env, num_entities = self.create_env_for_context(context)

        # Environment already has temporal observer, returns observations directly
        obs = env.reset()
        context_vector = self.get_context_vector(context)
        obs_with_context = add_context_to_observation(obs, context_vector)

        episode_reward = 0
        episode_length = 0

        done = False
        while not done and episode_length < 1000:
            # Select action
            action = self.policy_net.get_action(obs_with_context, epsilon=epsilon)

            # Execute action
            next_obs, reward, done, _ = env.step(action)
            next_obs_with_context = add_context_to_observation(next_obs, context_vector)

            # Store transition
            self.replay_buffer.append({
                'state': obs_with_context.copy(),
                'action': action,
                'reward': reward,
                'next_state': next_obs_with_context.copy(),
                'done': done
            })

            # Train policy network
            if len(self.replay_buffer) >= self.batch_size:
                policy_loss = self._train_policy_step()
                self.policy_losses.append(policy_loss)

                # Train world model
                world_model_loss = self._train_world_model_step()
                self.world_model_losses.append(world_model_loss)

            # Update target network
            if self.steps_done % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            obs_with_context = next_obs_with_context
            episode_reward += reward
            episode_length += 1
            self.steps_done += 1

        return episode_reward, episode_length

    def _train_policy_step(self):
        """Single policy network training step"""
        # Sample batch
        batch = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        transitions = [self.replay_buffer[i] for i in batch]

        # Convert to numpy arrays first, then to tensors (much faster!)
        states = torch.FloatTensor(np.stack([t['state'] for t in transitions]))
        actions = torch.LongTensor(np.array([t['action'] for t in transitions]))
        rewards = torch.FloatTensor(np.array([t['reward'] for t in transitions]))
        next_states = torch.FloatTensor(np.stack([t['next_state'] for t in transitions]))
        dones = torch.FloatTensor(np.array([t['done'] for t in transitions]))

        # Current Q-values
        current_q = self.policy_net.get_combined_q(states)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_net.get_combined_q(next_states)
            max_next_q = next_q.max(dim=1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # Compute loss
        loss = torch.nn.functional.mse_loss(current_q, target_q)

        # Optimize
        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.policy_optimizer.step()

        return loss.item()

    def _train_world_model_step(self):
        """Single world model training step"""
        # Sample batch
        batch = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        transitions = [self.replay_buffer[i] for i in batch]

        # Convert to numpy arrays first, then to tensors (much faster!)
        states = torch.FloatTensor(np.stack([t['state'] for t in transitions]))
        actions = torch.LongTensor(np.array([t['action'] for t in transitions]))
        rewards = torch.FloatTensor(np.array([t['reward'] for t in transitions]))
        next_states = torch.FloatTensor(np.stack([t['next_state'] for t in transitions]))
        dones = torch.FloatTensor(np.array([t['done'] for t in transitions]))

        # Forward pass
        pred_next_states, pred_rewards, pred_dones = self.world_model(states, actions)

        # Compute losses
        state_loss = torch.nn.functional.mse_loss(pred_next_states, next_states)
        reward_loss = torch.nn.functional.mse_loss(pred_rewards.squeeze(), rewards)
        done_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_dones.squeeze(), dones)

        total_loss = state_loss + reward_loss + done_loss

        # Optimize
        self.world_model_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.world_model_optimizer.step()

        return total_loss.item()

    def train(self, num_episodes, log_every=100):
        """Main training loop"""
        print("=" * 60)
        print("STARTING CONTEXT-AWARE TRAINING")
        print("=" * 60)
        print()

        best_avg_reward = -float('inf')

        for episode in range(num_episodes):
            # Sample context
            context = self.sample_context()
            self.context_episode_counts[context] += 1

            # Epsilon decay
            epsilon = max(0.01, 1.0 - episode / (num_episodes * 0.5))

            # Train episode
            reward, length = self.train_episode(context, epsilon)

            # Track stats
            self.episode_rewards.append(reward)
            self.episode_lengths.append(length)
            self.context_avg_rewards[context].append(reward)

            # Logging
            if (episode + 1) % log_every == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                avg_policy_loss = np.mean(self.policy_losses[-100:]) if self.policy_losses else 0
                avg_wm_loss = np.mean(self.world_model_losses[-100:]) if self.world_model_losses else 0

                print(f"Episode {episode+1}/{num_episodes}")
                print(f"  Avg Reward (100): {avg_reward:.2f}")
                print(f"  Avg Length (100): {avg_length:.1f}")
                print(f"  Policy Loss: {avg_policy_loss:.4f}")
                print(f"  World Model Loss: {avg_wm_loss:.4f}")
                print(f"  Epsilon: {epsilon:.3f}")
                print(f"  Buffer Size: {len(self.replay_buffer)}")
                print(f"  Steps: {self.steps_done}")

                # Context breakdown
                print("  Context Distribution:")
                for ctx in ['snake', 'balanced', 'survival']:
                    count = self.context_episode_counts[ctx]
                    pct = (count / (episode + 1)) * 100
                    ctx_avg = np.mean(self.context_avg_rewards[ctx][-50:]) if self.context_avg_rewards[ctx] else 0
                    print(f"    {ctx:8s}: {count:4d} episodes ({pct:4.1f}%) - avg reward: {ctx_avg:6.2f}")
                print()

                # Save best model
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.save(f"checkpoints/context_aware_{timestamp}_best")
                    print(f"  [BEST] Saved model (avg reward: {avg_reward:.2f})")
                    print()

        print("=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total episodes: {num_episodes}")
        print(f"Best avg reward: {best_avg_reward:.2f}")
        print(f"Final epsilon: {epsilon:.3f}")
        print()

        # Final context summary
        print("FINAL CONTEXT PERFORMANCE:")
        for ctx in ['snake', 'balanced', 'survival']:
            count = self.context_episode_counts[ctx]
            avg = np.mean(self.context_avg_rewards[ctx][-100:]) if len(self.context_avg_rewards[ctx]) >= 100 else np.mean(self.context_avg_rewards[ctx])
            print(f"  {ctx:8s}: {count:4d} episodes - avg reward: {avg:6.2f}")

    def save(self, base_path):
        """Save models"""
        os.makedirs(os.path.dirname(base_path), exist_ok=True)

        # Save policy
        policy_checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.policy_optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'context_episode_counts': self.context_episode_counts,
            'context_avg_rewards': self.context_avg_rewards,
            'steps_done': self.steps_done
        }
        torch.save(policy_checkpoint, f"{base_path}_policy.pth")

        # Save world model
        world_model_checkpoint = {
            'model': self.world_model.state_dict(),
            'optimizer': self.world_model_optimizer.state_dict(),
            'losses': self.world_model_losses
        }
        torch.save(world_model_checkpoint, f"{base_path}_world_model.pth")

        print(f"Saved checkpoint: {base_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Context-Aware Agent')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of episodes')
    parser.add_argument('--log-every', type=int, default=100, help='Log frequency')
    parser.add_argument('--env-size', type=int, default=20, help='Environment size')
    parser.add_argument('--num-rewards', type=int, default=10, help='Number of rewards (sparse for target-seeking)')

    args = parser.parse_args()

    print("=" * 60)
    print("CONTEXT-AWARE AGENT TRAINING")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Log every: {args.log_every}")
    print(f"Environment: {args.env_size}x{args.env_size}")
    print(f"Rewards: {args.num_rewards}")
    print()

    trainer = ContextAwareTrainer(
        env_size=args.env_size,
        num_rewards=args.num_rewards
    )

    trainer.train(
        num_episodes=args.episodes,
        log_every=args.log_every
    )

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer.save(f"checkpoints/context_aware_{timestamp}_final")


if __name__ == '__main__':
    main()
