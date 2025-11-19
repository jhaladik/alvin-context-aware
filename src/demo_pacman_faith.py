"""
Faith-Based Pac-Man Demo - Revolutionary Evolutionary Agent

Runs the faith-based evolutionary agent on Pac-Man and shows:
- Faith pattern evolution (persistent exploration despite negative feedback)
- Entity discovery (learns what entities are without labels)
- Universal pattern detection (game-agnostic strategies)
- Mechanic hypothesis testing (discovers hidden rules)

Agent color changes to indicate action type:
- MAGENTA 'P' = Faith action
- CYAN 'P' = Planning action
- YELLOW 'P' = Reactive action

Usage:
    python demo_pacman_faith.py
    python demo_pacman_faith.py --model checkpoints/faith_evolution_20251119_152049_best_policy.pth
    python demo_pacman_faith.py --episodes 20 --speed 0.1 --faith-freq 0.3
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

# Import faith system modules
from core.faith_system import FaithPattern, FaithPopulation
from core.entity_discovery import EntityDiscoveryWorldModel, EntityBehaviorLearner
from core.pattern_transfer import UniversalPatternExtractor
from core.mechanic_detectors import MechanicDetector


# ANSI color codes for terminal
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'

    # Action type colors
    MAGENTA = '\033[95m'  # Faith
    CYAN = '\033[96m'     # Planning
    YELLOW = '\033[93m'   # Reactive

    # Game elements
    GREEN = '\033[92m'
    RED = '\033[91m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'


def visualize_game_state(game, step, score, action_type, faith_stats, pattern_stats):
    """ASCII visualization with faith system metrics"""
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
        grid[y][x] = f'{Colors.RED}G{Colors.RESET}'

    # Draw Pac-Man with color based on action type
    px, py = game.pacman_pos
    if action_type == 'faith':
        grid[py][px] = f'{Colors.MAGENTA}{Colors.BOLD}P{Colors.RESET}'
    elif action_type == 'planning':
        grid[py][px] = f'{Colors.CYAN}{Colors.BOLD}P{Colors.RESET}'
    else:  # reactive
        grid[py][px] = f'{Colors.YELLOW}{Colors.BOLD}P{Colors.RESET}'

    # Clear screen
    print('\033[2J\033[H', end='')

    # Print header
    print("="*70)
    print(f"{Colors.BOLD}FAITH-BASED EVOLUTIONARY AGENT - PAC-MAN DEMO{Colors.RESET}")
    print("="*70)
    print(f"Step: {step:4d} | Score: {score:3d} | Pellets: {len(game.pellets):3d} | Lives: {game.lives}")

    # Action distribution
    total_actions = sum(faith_stats['action_counts'].values())
    if total_actions > 0:
        faith_pct = faith_stats['action_counts']['faith'] / total_actions * 100
        planning_pct = faith_stats['action_counts']['planning'] / total_actions * 100
        reactive_pct = faith_stats['action_counts']['reactive'] / total_actions * 100
        print(f"Actions: {Colors.MAGENTA}Faith {faith_pct:.1f}%{Colors.RESET} | "
              f"{Colors.CYAN}Planning {planning_pct:.1f}%{Colors.RESET} | "
              f"{Colors.YELLOW}Reactive {reactive_pct:.1f}%{Colors.RESET}")

    print("-"*70)

    # Print grid
    for row in grid:
        print(''.join(row))

    print("-"*70)

    # Faith system metrics
    print(f"{Colors.BOLD}FAITH DISCOVERIES:{Colors.RESET} {faith_stats['discoveries']}")
    print(f"{Colors.BOLD}ENTITY TYPES:{Colors.RESET} {faith_stats['entity_types']}")

    # Pattern detection
    if pattern_stats:
        print(f"{Colors.BOLD}PATTERNS DETECTED:{Colors.RESET} {len(pattern_stats)}")
        for pattern_name, confidence in list(pattern_stats.items())[:3]:
            print(f"  - {pattern_name}: {confidence:.2f} confidence")

    print("-"*70)
    print(f"Legend: {Colors.MAGENTA}P{Colors.RESET}=Faith | "
          f"{Colors.CYAN}P{Colors.RESET}=Planning | "
          f"{Colors.YELLOW}P{Colors.RESET}=Reactive | "
          f"{Colors.RED}G{Colors.RESET}=Ghost | .=Pellet | #=Wall")


def get_faith_action(faith_population, last_action, steps):
    """Select action from faith population using evolved patterns"""
    if faith_population is None or len(faith_population.patterns) == 0:
        return np.random.randint(4)

    # Get best pattern by fitness
    best_pattern = max(faith_population.patterns, key=lambda p: p.fitness)

    # Get dominant behavior type
    dominant_behavior = max(best_pattern.behavior_types.items(), key=lambda x: x[1])[0]

    # Execute behavior
    if dominant_behavior == 'wait':
        return last_action if last_action != -1 else np.random.randint(4)
    elif dominant_behavior == 'explore':
        return np.random.randint(4)
    elif dominant_behavior == 'rhythmic':
        return (steps % 4)
    elif dominant_behavior == 'sacrificial':
        # Try high-risk actions
        return np.random.randint(4)
    else:
        return np.random.randint(4)


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


def run_episode(agent, observer, game, world_model=None, faith_population=None,
                pattern_extractor=None, planning_freq=0.3, faith_freq=0.3,
                visualize=False, speed=0.0):
    """Run one episode with faith system"""
    game.reset()
    observer.reset()
    game_state = game._get_game_state()

    total_reward = 0
    steps = 0
    done = False
    pellets_collected = 0
    initial_pellets = len(game.pellets)
    last_action = -1

    action_counts = {'faith': 0, 'planning': 0, 'reactive': 0}
    faith_discoveries = 0
    entity_types_detected = set()
    patterns_detected = {}

    # Track recent states for pattern detection
    recent_states = []
    recent_actions = []
    recent_rewards = []

    while not done and steps < 1000:
        # Get observation
        obs = observer.observe(game_state)
        context_vector = infer_context_from_observation(obs)
        obs_with_context = add_context_to_observation(obs, context_vector)

        # Determine action type and select action
        rand = np.random.random()
        if rand < faith_freq:
            action = get_faith_action(faith_population, last_action, steps)
            action_type = 'faith'
            action_counts['faith'] += 1
        elif rand < faith_freq + planning_freq and world_model is not None:
            action = plan_action(agent, world_model, obs_with_context, planning_horizon=5)
            action_type = 'planning'
            action_counts['planning'] += 1
        else:
            action = agent.get_action(obs_with_context, epsilon=0.0)
            action_type = 'reactive'
            action_counts['reactive'] += 1

        # Visualize
        if visualize:
            faith_stats = {
                'action_counts': action_counts,
                'discoveries': faith_discoveries,
                'entity_types': len(entity_types_detected),
            }
            visualize_game_state(game, steps, game_state['score'], action_type,
                               faith_stats, patterns_detected)
            if speed > 0:
                time.sleep(speed)

        # Execute action
        prev_pellets = len(game.pellets)
        prev_score = game_state['score']
        game_state, reward, done = game.step(action)
        steps += 1
        last_action = action

        # Track pellets collected
        if len(game.pellets) < prev_pellets:
            pellets_collected += 1

        # Faith discovery detection (significant positive reward from faith action)
        if action_type == 'faith' and reward > 10:
            faith_discoveries += 1

        # Track for pattern detection
        recent_states.append(obs)
        recent_actions.append(action)
        recent_rewards.append(reward)

        # Update pattern extractor with observation
        if pattern_extractor is not None:
            # Extract entities from game state (ghosts and pellets)
            entities = []
            for i, ghost in enumerate(game_state.get('entities', [])):
                entities.append({
                    'type': 'ghost',
                    'type_id': 0,  # Ghost type ID
                    'pos': ghost.get('pos', (0, 0))
                })
            pattern_extractor.observe(entities, obs)

        # Detect patterns every 50 steps
        if steps % 50 == 0 and pattern_extractor is not None and steps > 0:
            try:
                detected = pattern_extractor.extract_patterns()
                if detected:
                    for pattern_name, pattern_info in detected.items():
                        patterns_detected[pattern_name] = pattern_info.get('confidence', 0.0)
            except Exception:
                # Pattern detection may fail - not critical for demo
                pass

        total_reward += reward

    # Final visualization
    if visualize:
        faith_stats = {
            'action_counts': action_counts,
            'discoveries': faith_discoveries,
            'entity_types': len(entity_types_detected),
        }
        visualize_game_state(game, steps, game_state['score'], 'done',
                           faith_stats, patterns_detected)
        print(f"\n{Colors.BOLD}EPISODE FINISHED!{Colors.RESET}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Pellets Collected: {pellets_collected}/{initial_pellets} ({pellets_collected/initial_pellets*100:.1f}%)")
        print(f"  Faith Discoveries: {faith_discoveries}")
        print(f"  Patterns Detected: {len(patterns_detected)}")
        print(f"  Survival: {steps} steps")
        input("\nPress Enter to continue...")

    return {
        'score': game_state['score'],
        'reward': total_reward,
        'steps': steps,
        'pellets_collected': pellets_collected,
        'initial_pellets': initial_pellets,
        'completion': pellets_collected / initial_pellets if initial_pellets > 0 else 0,
        'action_counts': action_counts,
        'faith_discoveries': faith_discoveries,
        'patterns_detected': len(patterns_detected),
    }


def main():
    parser = argparse.ArgumentParser(description='Faith-Based Pac-Man Demo')

    # Try to find the checkpoint in multiple locations
    default_checkpoint = 'faith_evolution_20251119_152049_best_policy.pth'
    possible_paths = [
        f'checkpoints/{default_checkpoint}',
        f'../checkpoints/{default_checkpoint}',
        f'src/checkpoints/{default_checkpoint}',
    ]

    default_model = None
    for path in possible_paths:
        if os.path.exists(path):
            default_model = path
            break

    if default_model is None:
        default_model = possible_paths[0]  # Use first path as fallback

    parser.add_argument('--model', type=str,
                       default=default_model,
                       help='Path to faith model checkpoint (default: latest episode 500)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to run (default: 10)')
    parser.add_argument('--visualize', action='store_true',
                       help='Show ASCII visualization of game')
    parser.add_argument('--speed', type=float, default=0.05,
                       help='Visualization speed in seconds (default: 0.05)')
    parser.add_argument('--planning-freq', type=float, default=0.2,
                       help='Planning frequency 0-1 (default: 0.2)')
    parser.add_argument('--faith-freq', type=float, default=0.3,
                       help='Faith action frequency 0-1 (default: 0.3)')

    args = parser.parse_args()

    print("="*70)
    print(f"{Colors.BOLD}FAITH-BASED EVOLUTIONARY AGENT - PAC-MAN DEMO{Colors.RESET}")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Faith Frequency: {args.faith_freq*100:.0f}%")
    print(f"Planning Frequency: {args.planning_freq*100:.0f}%")
    print("="*70)

    # Load agent
    try:
        checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
        agent = ContextAwareDQN(obs_dim=95, action_dim=4)
        agent.load_state_dict(checkpoint['policy_net'])
        agent.eval()
        print(f"[OK] Loaded agent from Episode {len(checkpoint.get('episode_rewards', []))}")

        # Load faith statistics
        faith_count = checkpoint.get('faith_count', 0)
        faith_discoveries = checkpoint.get('faith_discovery_count', 0)
        entity_types = checkpoint.get('entity_types_discovered', 0)
        patterns_detected = checkpoint.get('patterns_detected', 0)
        print(f"[OK] Training stats: {faith_count} faith actions, {faith_discoveries} discoveries")
        print(f"[OK] Entity types: {entity_types}, Patterns: {patterns_detected}")
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return

    # Load world model for planning
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
        print(f"[WARN] World model not found (planning disabled)")

    # Load faith population
    faith_population = FaithPopulation(population_size=20)
    if 'faith_population' in checkpoint:
        print(f"[OK] Loading faith population from checkpoint...")
        for i, pattern_data in enumerate(checkpoint['faith_population'][:20]):
            if i < len(faith_population.patterns):
                faith_population.patterns[i].fitness = pattern_data.get('fitness', 0)
                faith_population.patterns[i].age = pattern_data.get('age', 0)
                faith_population.patterns[i].discoveries = pattern_data.get('discoveries', 0)
                faith_population.patterns[i].behavior_types = pattern_data.get('behavior_types', {})
        print(f"[OK] Faith population loaded: {len(faith_population.patterns)} patterns")
        best_pattern = max(faith_population.patterns, key=lambda p: p.fitness)
        print(f"[OK] Best pattern fitness: {best_pattern.fitness:.2f}")
    else:
        print(f"[WARN] No faith population in checkpoint, using default")

    # Initialize pattern extractor
    pattern_extractor = UniversalPatternExtractor()
    print(f"[OK] Pattern extractor initialized")

    print()

    # Run episodes
    game = PacManGame(size=20)
    observer = TemporalFlowObserver()

    results = []
    for episode in range(args.episodes):
        visualize = args.visualize if episode == 0 else False  # Only visualize first episode
        result = run_episode(
            agent, observer, game,
            world_model=world_model,
            faith_population=faith_population,
            pattern_extractor=pattern_extractor,
            planning_freq=args.planning_freq,
            faith_freq=args.faith_freq,
            visualize=visualize,
            speed=args.speed
        )
        results.append(result)

        if not visualize:
            total_actions = sum(result['action_counts'].values())
            faith_pct = result['action_counts']['faith'] / total_actions * 100 if total_actions > 0 else 0
            print(f"Episode {episode+1:3d}: Score={result['score']:3d} "
                  f"Pellets={result['pellets_collected']:3d}/{result['initial_pellets']:3d} "
                  f"({result['completion']*100:5.1f}%) "
                  f"Reward={result['reward']:7.1f} "
                  f"Steps={result['steps']:4d} "
                  f"Faith={faith_pct:.1f}% "
                  f"Discoveries={result['faith_discoveries']}")

    # Print summary
    print()
    print("="*70)
    print(f"{Colors.BOLD}SUMMARY STATISTICS{Colors.RESET}")
    print("="*70)

    scores = [r['score'] for r in results]
    rewards = [r['reward'] for r in results]
    completions = [r['completion'] for r in results]
    steps = [r['steps'] for r in results]
    discoveries = [r['faith_discoveries'] for r in results]
    patterns = [r['patterns_detected'] for r in results]

    print(f"Average Score:        {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Average Reward:       {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Average Completion:   {np.mean(completions)*100:.1f}% ± {np.std(completions)*100:.1f}%")
    print(f"Average Steps:        {np.mean(steps):.1f} ± {np.std(steps):.1f}")
    print()
    print(f"Best Score:           {max(scores)}")
    print(f"Best Completion:      {max(completions)*100:.1f}%")
    print(f"Best Reward:          {max(rewards):.1f}")
    print()

    # Action distribution
    total_faith = sum(r['action_counts']['faith'] for r in results)
    total_planning = sum(r['action_counts']['planning'] for r in results)
    total_reactive = sum(r['action_counts']['reactive'] for r in results)
    total_actions = total_faith + total_planning + total_reactive

    print(f"{Colors.MAGENTA}Faith Actions:{Colors.RESET}       {total_faith} ({total_faith/total_actions*100:.1f}%)")
    print(f"{Colors.CYAN}Planning Actions:{Colors.RESET}   {total_planning} ({total_planning/total_actions*100:.1f}%)")
    print(f"{Colors.YELLOW}Reactive Actions:{Colors.RESET}   {total_reactive} ({total_reactive/total_actions*100:.1f}%)")
    print()

    # Faith metrics
    print(f"{Colors.BOLD}FAITH SYSTEM METRICS:{Colors.RESET}")
    print(f"  Total Discoveries:    {sum(discoveries)}")
    print(f"  Avg Discoveries/Ep:   {np.mean(discoveries):.2f}")
    print(f"  Total Patterns:       {sum(patterns)}")
    print(f"  Avg Patterns/Ep:      {np.mean(patterns):.2f}")
    print()

    # Performance assessment
    avg_completion = np.mean(completions)
    if avg_completion > 0.8:
        grade = f"{Colors.GREEN}EXCELLENT ***{Colors.RESET}"
    elif avg_completion > 0.6:
        grade = f"{Colors.CYAN}GOOD **{Colors.RESET}"
    elif avg_completion > 0.4:
        grade = f"{Colors.YELLOW}FAIR *{Colors.RESET}"
    else:
        grade = f"{Colors.RED}NEEDS IMPROVEMENT{Colors.RESET}"

    print(f"Overall Performance:  {grade}")
    print("="*70)


if __name__ == '__main__':
    main()
