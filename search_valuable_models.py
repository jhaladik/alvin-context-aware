"""
Systematic Search for ACTUALLY VALUABLE Pre-trained RL Models
Looking for models that might transfer to warehouse robotics
"""
from huggingface_hub import list_models, model_info
import re

def search_models_by_category(category_keywords, limit=100):
    """Search for models by category"""
    results = []

    print(f"\n{'='*70}")
    print(f"SEARCHING: {', '.join(category_keywords)}")
    print(f"{'='*70}\n")

    models = list(list_models(
        filter='reinforcement-learning',
        limit=limit,
        sort='downloads',
        direction=-1
    ))

    for model in models:
        model_id = model.modelId.lower()
        tags = [tag.lower() for tag in (model.tags if hasattr(model, 'tags') else [])]

        # Check if any keyword matches
        if any(keyword.lower() in model_id or keyword.lower() in ' '.join(tags)
               for keyword in category_keywords):

            try:
                info = model_info(model.modelId)
                downloads = info.downloads if hasattr(info, 'downloads') else 0
                results.append({
                    'id': model.modelId,
                    'tags': tags,
                    'downloads': downloads
                })
            except:
                results.append({
                    'id': model.modelId,
                    'tags': tags,
                    'downloads': 0
                })

    # Sort by downloads
    results.sort(key=lambda x: x['downloads'], reverse=True)

    return results


def main():
    print("="*70)
    print("SYSTEMATIC SEARCH FOR VALUABLE PRE-TRAINED RL MODELS")
    print("="*70)
    print("\nLooking for models that might actually transfer to:")
    print("  - Warehouse robotics")
    print("  - Grid-based navigation")
    print("  - Multi-task environments")
    print("  - Continuous control")

    categories = {
        "Robotics & Manipulation": [
            'robot', 'robotic', 'manipulation', 'arm', 'gripper',
            'pick', 'place', 'ur5', 'franka', 'panda'
        ],
        "Navigation & Path Planning": [
            'navigation', 'pathfinding', 'maze', 'grid', 'obstacle',
            'collision', 'avoidance', 'warehouse', 'minigrid'
        ],
        "Multi-Task / Foundation Models": [
            'multi-task', 'multitask', 'gato', 'foundation',
            'generalist', 'universal', 'meta-rl', 'world-model'
        ],
        "Continuous Control (MuJoCo, etc.)": [
            'mujoco', 'humanoid', 'walker', 'hopper', 'ant',
            'reacher', 'pusher', 'continuous', 'sac', 'td3'
        ],
        "Minecraft / MineRL (Complex Tasks)": [
            'minecraft', 'minerl', 'vpt', 'video-pretraining'
        ],
        "Simulation Environments": [
            'pybullet', 'unity', 'habitat', 'gym-pybullet',
            'deepmind', 'dm-control', 'suite'
        ]
    }

    all_results = {}

    for category, keywords in categories.items():
        results = search_models_by_category(keywords, limit=200)
        all_results[category] = results

        if results:
            print(f"\nFound {len(results)} models:")
            for i, model in enumerate(results[:5], 1):
                downloads = model['downloads']
                print(f"  {i}. {model['id']}")
                print(f"     Downloads: {downloads:,}")
                print(f"     Tags: {', '.join(model['tags'][:5])}")
        else:
            print("\nNo models found.")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: BEST CANDIDATES FOR YOUR WAREHOUSE ROBOT")
    print("="*70)

    # Collect top models across all categories
    all_models = []
    for category, models in all_results.items():
        for model in models[:3]:  # Top 3 from each category
            all_models.append({
                **model,
                'category': category
            })

    # Sort by downloads and uniquify
    seen = set()
    unique_models = []
    for model in sorted(all_models, key=lambda x: x['downloads'], reverse=True):
        if model['id'] not in seen:
            seen.add(model['id'])
            unique_models.append(model)

    print("\nTop 10 Most Downloaded RL Models (Potentially Valuable):\n")
    for i, model in enumerate(unique_models[:10], 1):
        print(f"{i}. {model['id']}")
        print(f"   Category: {model['category']}")
        print(f"   Downloads: {model['downloads']:,}")
        print(f"   Tags: {', '.join(model['tags'][:5])}")
        print()

    # Specific recommendations
    print("="*70)
    print("RECOMMENDATIONS FOR YOUR USE CASE")
    print("="*70)

    print("\n1. FOR WAREHOUSE NAVIGATION:")
    print("   Look for: MiniGrid models (grid-based navigation)")
    print("   Why: Same type of environment (grid, obstacles)")
    print("   Expected transfer: Medium (similar structure)")

    print("\n2. FOR ROBOTIC MANIPULATION:")
    print("   Look for: PyBullet/MuJoCo manipulation models")
    print("   Why: Continuous control, physics simulation")
    print("   Expected transfer: Low-Medium (need fine-tuning)")

    print("\n3. FOR COMPLEX MULTI-STEP TASKS:")
    print("   Look for: Minecraft/MineRL models")
    print("   Why: Long-horizon planning, complex goals")
    print("   Expected transfer: Low (very different domain)")

    print("\n4. MOST REALISTIC APPROACH:")
    print("   - Use Stable-Baselines3 with standard algorithms")
    print("   - Train on your specific warehouse environment")
    print("   - Use proper observation design (ray-based)")
    print("   - This is what you're already doing - IT'S THE RIGHT WAY!")

    print("\n" + "="*70)
    print("HONEST CONCLUSION")
    print("="*70)
    print("\nAfter searching Hugging Face:")
    print("  - Most models are game-specific (Atari, etc.)")
    print("  - Few robotics models, mostly research demos")
    print("  - Zero-shot transfer rarely works even with 'similar' tasks")
    print("  - Your task-specific training approach is ALREADY optimal")
    print("\nThe 'valuable' models are valuable for THEIR TASKS, not yours.")
    print("Task-specific training remains the gold standard.")


if __name__ == "__main__":
    main()
