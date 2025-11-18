"""
Compare Context-Aware Agent Checkpoints

Analyzes and compares training progress across multiple checkpoint files.
Helps identify best models and understand training dynamics.

Usage:
    python compare_checkpoints.py
    python compare_checkpoints.py --top 5
    python compare_checkpoints.py --detailed ../checkpoints/context_aware_20251118_115931_best_policy.pth
"""

import torch
import glob
import os
import argparse
import numpy as np
from datetime import datetime


def load_checkpoint_info(checkpoint_path):
    """Extract key information from checkpoint"""
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')

        # Extract metrics
        episode_rewards = ckpt.get('episode_rewards', [])
        episode_lengths = ckpt.get('episode_lengths', [])
        context_counts = ckpt.get('context_episode_counts', {})
        context_rewards = ckpt.get('context_avg_rewards', {})
        steps = ckpt.get('steps_done', 0)

        # Compute statistics
        num_episodes = len(episode_rewards)

        if num_episodes == 0:
            return None

        # Last 100 episodes average (or all if less)
        window = min(100, num_episodes)
        recent_avg_reward = np.mean(episode_rewards[-window:])
        recent_avg_length = np.mean(episode_lengths[-window:])

        # Overall statistics
        total_reward = sum(episode_rewards)
        max_reward = max(episode_rewards)
        min_reward = min(episode_rewards)

        # Context-specific performance
        context_performance = {}
        for ctx in ['snake', 'balanced', 'survival']:
            ctx_rewards = context_rewards.get(ctx, [])
            if ctx_rewards:
                context_performance[ctx] = {
                    'count': context_counts.get(ctx, 0),
                    'avg_reward': np.mean(ctx_rewards),
                    'recent_avg': np.mean(ctx_rewards[-50:]) if len(ctx_rewards) >= 50 else np.mean(ctx_rewards)
                }
            else:
                context_performance[ctx] = {
                    'count': 0,
                    'avg_reward': 0,
                    'recent_avg': 0
                }

        # Extract timestamp from filename
        filename = os.path.basename(checkpoint_path)
        try:
            # Format: context_aware_YYYYMMDD_HHMMSS_best_policy.pth
            timestamp_str = filename.split('_')[2] + '_' + filename.split('_')[3]
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except:
            timestamp = None

        return {
            'path': checkpoint_path,
            'filename': filename,
            'timestamp': timestamp,
            'num_episodes': num_episodes,
            'steps': steps,
            'recent_avg_reward': recent_avg_reward,
            'recent_avg_length': recent_avg_length,
            'total_reward': total_reward,
            'max_reward': max_reward,
            'min_reward': min_reward,
            'context_performance': context_performance,
            'all_rewards': episode_rewards,
            'all_lengths': episode_lengths
        }
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None


def print_checkpoint_summary(info, rank=None):
    """Print summary for a single checkpoint"""
    if rank:
        print(f"\n{'='*80}")
        print(f"RANK #{rank}")
    print(f"{'='*80}")
    print(f"File: {info['filename']}")
    if info['timestamp']:
        print(f"Timestamp: {info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nTRAINING PROGRESS:")
    print(f"  Episodes:          {info['num_episodes']:,}")
    print(f"  Total Steps:       {info['steps']:,}")
    print(f"  Avg Reward (100):  {info['recent_avg_reward']:.2f}")
    print(f"  Avg Length (100):  {info['recent_avg_length']:.1f}")
    print(f"  Max Reward:        {info['max_reward']:.2f}")
    print(f"  Min Reward:        {info['min_reward']:.2f}")

    print(f"\nCONTEXT PERFORMANCE:")
    total_episodes = sum(ctx['count'] for ctx in info['context_performance'].values())
    for ctx_name in ['snake', 'balanced', 'survival']:
        ctx = info['context_performance'][ctx_name]
        pct = (ctx['count'] / total_episodes * 100) if total_episodes > 0 else 0
        print(f"  {ctx_name:8s}: {ctx['count']:4d} episodes ({pct:5.1f}%) - "
              f"avg: {ctx['avg_reward']:6.2f}, recent: {ctx['recent_avg']:6.2f}")


def print_detailed_analysis(info):
    """Print detailed analysis including training curves"""
    print_checkpoint_summary(info)

    print(f"\nTRAINING CURVE (every {max(1, info['num_episodes']//20)} episodes):")
    print(f"  {'Episode':>8s} | {'Avg Reward':>12s} | {'Avg Length':>12s}")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*12}")

    rewards = info['all_rewards']
    lengths = info['all_lengths']
    step_size = max(1, info['num_episodes'] // 20)

    for i in range(0, info['num_episodes'], step_size):
        window_start = max(0, i - 50)
        window_end = min(len(rewards), i + 50)
        avg_r = np.mean(rewards[window_start:window_end])
        avg_l = np.mean(lengths[window_start:window_end])
        print(f"  {i:8d} | {avg_r:12.2f} | {avg_l:12.1f}")

    # Final stats
    print(f"\nLEARNING DYNAMICS:")
    if info['num_episodes'] >= 100:
        early_reward = np.mean(rewards[:100])
        late_reward = np.mean(rewards[-100:])
        improvement = late_reward - early_reward
        print(f"  Early performance (first 100):  {early_reward:6.2f}")
        print(f"  Late performance (last 100):    {late_reward:6.2f}")
        print(f"  Improvement:                    {improvement:+6.2f} ({improvement/early_reward*100:+.1f}%)")

    # Stability analysis
    if info['num_episodes'] >= 50:
        recent_std = np.std(rewards[-50:])
        print(f"  Recent stability (std last 50): {recent_std:6.2f}")


def compare_all_checkpoints(checkpoint_dir='../checkpoints', top_n=None):
    """Compare all checkpoints and rank them"""
    pattern = os.path.join(checkpoint_dir, '*_best_policy.pth')
    checkpoint_files = sorted(glob.glob(pattern))

    if not checkpoint_files:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    print(f"Found {len(checkpoint_files)} checkpoint files")
    print("Loading checkpoint information...")

    # Load all checkpoint info
    checkpoints = []
    for ckpt_file in checkpoint_files:
        info = load_checkpoint_info(ckpt_file)
        if info:
            checkpoints.append(info)

    if not checkpoints:
        print("No valid checkpoints found")
        return

    # Sort by recent average reward (descending)
    checkpoints.sort(key=lambda x: x['recent_avg_reward'], reverse=True)

    print(f"\n{'='*80}")
    print("CHECKPOINT COMPARISON - RANKED BY RECENT AVERAGE REWARD (last 100 episodes)")
    print(f"{'='*80}")
    print(f"\n{'Rank':>4s} | {'Episodes':>8s} | {'Steps':>8s} | {'Avg(100)':>10s} | {'Max':>8s} | {'File'}")
    print(f"{'-'*4}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*40}")

    display_count = top_n if top_n else len(checkpoints)

    for i, info in enumerate(checkpoints[:display_count], 1):
        print(f"{i:4d} | {info['num_episodes']:8d} | {info['steps']:8d} | "
              f"{info['recent_avg_reward']:10.2f} | {info['max_reward']:8.2f} | "
              f"{info['filename'][:60]}")

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")

    best = checkpoints[0]
    most_trained = max(checkpoints, key=lambda x: x['num_episodes'])
    most_steps = max(checkpoints, key=lambda x: x['steps'])

    print(f"\nBest Performance (Avg Reward):")
    print(f"  {best['filename']}")
    print(f"  Episodes: {best['num_episodes']}, Avg Reward: {best['recent_avg_reward']:.2f}")

    print(f"\nMost Episodes Trained:")
    print(f"  {most_trained['filename']}")
    print(f"  Episodes: {most_trained['num_episodes']}, Avg Reward: {most_trained['recent_avg_reward']:.2f}")

    print(f"\nMost Steps:")
    print(f"  {most_steps['filename']}")
    print(f"  Steps: {most_steps['steps']:,}, Avg Reward: {most_steps['recent_avg_reward']:.2f}")

    # Context distribution across all checkpoints
    print(f"\n{'='*80}")
    print("CONTEXT DISTRIBUTION ANALYSIS")
    print(f"{'='*80}")

    total_snake = sum(c['context_performance']['snake']['count'] for c in checkpoints)
    total_balanced = sum(c['context_performance']['balanced']['count'] for c in checkpoints)
    total_survival = sum(c['context_performance']['survival']['count'] for c in checkpoints)
    total_all = total_snake + total_balanced + total_survival

    if total_all > 0:
        print(f"  Snake:    {total_snake:5d} episodes ({total_snake/total_all*100:5.1f}%) - target: 30.0%")
        print(f"  Balanced: {total_balanced:5d} episodes ({total_balanced/total_all*100:5.1f}%) - target: 50.0%")
        print(f"  Survival: {total_survival:5d} episodes ({total_survival/total_all*100:5.1f}%) - target: 20.0%")

    return checkpoints


def main():
    parser = argparse.ArgumentParser(description='Compare Context-Aware Agent Checkpoints')
    parser.add_argument('--checkpoint-dir', type=str, default='../checkpoints',
                        help='Directory containing checkpoint files')
    parser.add_argument('--top', type=int, default=None,
                        help='Show only top N checkpoints')
    parser.add_argument('--detailed', type=str, default=None,
                        help='Show detailed analysis for specific checkpoint')

    args = parser.parse_args()

    if args.detailed:
        # Detailed analysis of single checkpoint
        info = load_checkpoint_info(args.detailed)
        if info:
            print_detailed_analysis(info)
        else:
            print(f"Could not load checkpoint: {args.detailed}")
    else:
        # Compare all checkpoints
        checkpoints = compare_all_checkpoints(args.checkpoint_dir, args.top)

        if checkpoints and len(checkpoints) > 0:
            print(f"\n{'='*80}")
            print("RECOMMENDATION")
            print(f"{'='*80}")
            best = checkpoints[0]
            print(f"\nFor best performance, use:")
            print(f"  {best['path']}")
            print(f"\nTest with:")
            print(f"  python test_context_aware.py {best['path']} --episodes 50")
            print(f"\nVisualize with:")
            print(f"  python context_aware_visual_games.py --model {best['path']}")


if __name__ == '__main__':
    main()
