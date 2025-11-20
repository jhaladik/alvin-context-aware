"""
Complete Testing Workflow - Run All Comparisons and Applications

This script automates the complete testing process:
1. Identifies trained models (baseline vs expanded)
2. Runs architecture comparison on standard games
3. Tests both models on warehouse scenarios
4. Generates comprehensive reports

Usage:
    python run_complete_testing.py
    python run_complete_testing.py --episodes 50
    python run_complete_testing.py --quick  # 20 episodes for quick test
"""
import sys
import os
import argparse
import subprocess
from pathlib import Path


def find_latest_model(pattern):
    """Find latest checkpoint matching pattern"""
    checkpoints_dir = Path("checkpoints")
    matches = sorted(checkpoints_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def run_command(cmd, description):
    """Run command and print output"""
    print(f"\n{'='*100}")
    print(f"{description}")
    print(f"{'='*100}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n⚠️  Command failed with exit code {result.returncode}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description='Run complete testing workflow')
    parser.add_argument('--episodes', type=int, default=50, help='Episodes per test')
    parser.add_argument('--quick', action='store_true', help='Quick test (20 episodes)')
    parser.add_argument('--skip-comparison', action='store_true', help='Skip architecture comparison')
    parser.add_argument('--skip-warehouse', action='store_true', help='Skip warehouse scenarios')
    parser.add_argument('--baseline-model', type=str, help='Path to baseline model')
    parser.add_argument('--expanded-model', type=str, help='Path to expanded model')

    args = parser.parse_args()

    if args.quick:
        args.episodes = 20

    print(f"{'='*100}")
    print(f"COMPLETE TESTING WORKFLOW")
    print(f"{'='*100}")
    print(f"\nConfiguration:")
    print(f"  Episodes per test: {args.episodes}")
    print(f"  Skip comparison: {args.skip_comparison}")
    print(f"  Skip warehouse: {args.skip_warehouse}")

    # Step 1: Find models
    print(f"\n{'='*100}")
    print(f"STEP 1: LOCATING MODELS")
    print(f"{'='*100}")

    if args.baseline_model:
        baseline_model = Path(args.baseline_model)
    else:
        print("\nSearching for baseline model (context_aware_advanced)...")
        baseline_model = find_latest_model("context_aware_advanced_*_final_policy.pth")

        if not baseline_model:
            # Try best models
            baseline_model = find_latest_model("context_aware_advanced_*_best_policy.pth")

        if not baseline_model:
            # Fall back to any context_aware model
            baseline_model = find_latest_model("context_aware_*_final_policy.pth")

    if args.expanded_model:
        expanded_model = Path(args.expanded_model)
    else:
        print("Searching for expanded model (faith_evolution)...")
        expanded_model = find_latest_model("faith_evolution_*_final_policy.pth")

        if not expanded_model:
            # Try best models
            expanded_model = find_latest_model("faith_evolution_*_best_policy.pth")

    # Report findings
    if baseline_model and baseline_model.exists():
        print(f"\n✅ Baseline model found: {baseline_model}")
        print(f"   Size: {baseline_model.stat().st_size / 1024:.1f} KB")
    else:
        print(f"\n❌ Baseline model not found!")
        print(f"   Please specify with --baseline-model")
        return

    if expanded_model and expanded_model.exists():
        print(f"\n✅ Expanded model found: {expanded_model}")
        print(f"   Size: {expanded_model.stat().st_size / 1024:.1f} KB")
    else:
        print(f"\n❌ Expanded model not found!")
        print(f"   Please specify with --expanded-model")
        return

    # Step 2: Run architecture comparison
    if not args.skip_comparison:
        print(f"\n{'='*100}")
        print(f"STEP 2: ARCHITECTURE COMPARISON (Standard Games)")
        print(f"{'='*100}")

        comparison_cmd = [
            "python", "compare_model_architectures.py",
            "--baseline", str(baseline_model),
            "--expanded", str(expanded_model),
            "--episodes", str(args.episodes)
        ]

        success = run_command(comparison_cmd, "Running architecture comparison...")

        if not success:
            print("\n⚠️  Architecture comparison failed, continuing anyway...")

    # Step 3: Test warehouse scenarios
    if not args.skip_warehouse:
        print(f"\n{'='*100}")
        print(f"STEP 3: WAREHOUSE SCENARIOS")
        print(f"{'='*100}")

        # Test baseline model on warehouse
        print(f"\n--- Testing BASELINE model on warehouse scenarios ---")
        warehouse_baseline_cmd = [
            "python", "warehouse_faith_demo.py",
            str(baseline_model),
            "--scenario", "all",
            "--episodes", str(args.episodes)
        ]

        success = run_command(warehouse_baseline_cmd, "Testing baseline model on warehouse...")

        if not success:
            print("\n⚠️  Baseline warehouse test failed, continuing anyway...")

        # Test expanded model on warehouse
        print(f"\n--- Testing EXPANDED model on warehouse scenarios ---")
        warehouse_expanded_cmd = [
            "python", "warehouse_faith_demo.py",
            str(expanded_model),
            "--scenario", "all",
            "--episodes", str(args.episodes)
        ]

        success = run_command(warehouse_expanded_cmd, "Testing expanded model on warehouse...")

        if not success:
            print("\n⚠️  Expanded warehouse test failed, continuing anyway...")

    # Summary
    print(f"\n{'='*100}")
    print(f"TESTING WORKFLOW COMPLETE")
    print(f"{'='*100}")

    print(f"\nGenerated Files:")
    if os.path.exists("model_comparison_report.txt"):
        print(f"  ✅ model_comparison_report.txt - Architecture comparison results")

    print(f"\nModels Tested:")
    print(f"  Baseline: {baseline_model}")
    print(f"  Expanded: {expanded_model}")

    print(f"\nNext Steps:")
    print(f"  1. Review model_comparison_report.txt for performance differences")
    print(f"  2. Check warehouse scenario results in terminal output")
    print(f"  3. If expanded model performs better, use it for warehouse application")
    print(f"  4. If more training needed, continue with train_expanded_faith.py")

    print(f"\nRecommended Commands:")
    print(f"  # Review comparison report")
    print(f"  cat model_comparison_report.txt")
    print(f"")
    print(f"  # Continue training expanded model")
    print(f"  python train_expanded_faith.py --episodes 500 --resume {expanded_model}")
    print(f"")
    print(f"  # Test specific warehouse scenario")
    print(f"  python warehouse_faith_demo.py {expanded_model} --scenario hidden_shortcut --episodes 50")


if __name__ == '__main__':
    main()
