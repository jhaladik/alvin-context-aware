"""
Warehouse Faith Visual Interface

Interactive visualization for testing faith-based agents on warehouse scenarios.
Shows real-time discovery of hidden mechanics, faith action execution, and
scenario-specific information.

Usage:
    python warehouse_faith_visual.py --model <checkpoint> --scenario hidden_shortcut
    python warehouse_faith_visual.py --model <checkpoint> --scenario charging_station --faith-freq 0.3
"""

import pygame
import sys
import os
import torch
import numpy as np
import argparse

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))

from warehouse_faith_scenarios import (
    create_scenario,
    get_scenario_descriptions,
    FLOOR_COLOR, SHELF_COLOR, AGV_COLOR, WORKER_COLOR, PACKAGE_COLOR,
    CHARGING_STATION_COLOR, SUPERVISOR_COLOR,
    PRIORITY_RED, PRIORITY_BLUE, PRIORITY_GREEN,
    GREEN, YELLOW, RED, ORANGE, MAGENTA, WHITE, BLACK, GRAY
)

from context_aware_agent import (
    ContextAwareDQN,
    infer_context_from_observation,
    add_context_to_observation
)
from core.temporal_observer import TemporalFlowObserver
from core.world_model import WorldModelNetwork
from core.faith_system import FaithPopulation


class WarehouseFaithVisualizer:
    """Interactive visualizer for warehouse faith scenarios"""

    def __init__(self, model_path=None, scenario_name='hidden_shortcut',
                 cell_size=30, faith_freq=0.3, use_planning=True,
                 planning_freq=0.2, planning_horizon=5):

        self.cell_size = cell_size
        self.faith_freq = faith_freq
        self.use_planning = use_planning
        self.planning_freq = planning_freq
        self.planning_horizon = planning_horizon

        # Create scenario
        self.scenario = create_scenario(scenario_name, size=20)
        self.observer = TemporalFlowObserver()

        # Load agent
        self.agent = None
        self.world_model = None
        self.faith_population = None
        self.current_obs = None
        self.current_context_name = "Unknown"

        if model_path and os.path.exists(model_path):
            self._load_model(model_path)

        # Initialize pygame
        pygame.init()
        self.info_panel_width = 400
        self.screen_width = self.scenario.size * self.cell_size + self.info_panel_width
        self.screen_height = self.scenario.size * self.cell_size + 100
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption(f'Warehouse Faith - {self.scenario.name}')

        # Fonts
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.tiny_font = pygame.font.Font(None, 14)
        self.title_font = pygame.font.Font(None, 28)
        self.clock = pygame.time.Clock()

        # Tracking
        self.action_counts = {'faith': 0, 'planning': 0, 'reactive': 0}
        self.last_action_source = 'reactive'
        self.use_ai = True if self.agent else False

    def _load_model(self, model_path):
        """Load faith-based agent"""
        print(f"Loading model: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        # Load policy
        self.agent = ContextAwareDQN(obs_dim=95, action_dim=4)
        self.agent.load_state_dict(checkpoint['policy_net'])
        self.agent.eval()

        # Load world model
        base_path = model_path.replace('_policy.pth', '')
        world_model_path = f"{base_path}_world_model.pth"

        if self.use_planning and os.path.exists(world_model_path):
            self.world_model = WorldModelNetwork(state_dim=95, action_dim=4)
            wm_checkpoint = torch.load(world_model_path, map_location='cpu', weights_only=False)
            self.world_model.load_state_dict(wm_checkpoint['model'])
            self.world_model.eval()
            print(f"  Planning enabled")

        # Load faith population
        self.faith_population = FaithPopulation(population_size=20)
        if 'faith_population' in checkpoint:
            for i, pattern_data in enumerate(checkpoint['faith_population'][:20]):
                if i < len(self.faith_population.patterns):
                    self.faith_population.patterns[i].fitness = pattern_data.get('fitness', 0)
            print(f"  Faith population loaded: {len(self.faith_population.patterns)} patterns")

        print(f"Model loaded: {len(checkpoint.get('episode_rewards', []))} episodes trained")
        print(f"  Faith actions trained: {checkpoint.get('faith_count', 0)}")
        print(f"  Discoveries: {checkpoint.get('faith_discovery_count', 0)}")

    def _get_faith_action(self):
        """Select faith action"""
        if not self.faith_population:
            return np.random.randint(4)

        best_pattern = max(self.faith_population.patterns, key=lambda p: p.fitness)
        dominant_behavior = max(best_pattern.behavior_types.items(), key=lambda x: x[1])[0]

        if dominant_behavior == 'explore':
            return np.random.randint(4)
        elif dominant_behavior == 'rhythmic':
            return (self.scenario.steps % 4)
        else:
            return np.random.randint(4)

    def _plan_action(self, state):
        """Use world model for planning"""
        if not self.world_model:
            return np.random.randint(4)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        best_action = 0
        best_return = -float('inf')

        with torch.no_grad():
            for action in range(4):
                total_return = 0.0
                current_state = state_tensor.clone()

                # Simulate horizon
                for _ in range(self.planning_horizon):
                    action_tensor = torch.LongTensor([action])
                    next_state, reward, done = self.world_model(current_state, action_tensor)
                    total_return += reward.item()
                    if done.item() > 0.5:
                        break
                    current_state = next_state

                if total_return > best_return:
                    best_return = total_return
                    best_action = action

        return best_action

    def draw_scenario(self):
        """Draw warehouse scenario"""
        self.screen.fill(FLOOR_COLOR)

        # Draw walls/shelves
        for x, y in self.scenario.walls:
            color = SHELF_COLOR
            # Highlight shortcuts in hidden shortcut scenario
            if (hasattr(self.scenario, 'shortcut_walls') and
                (x, y) in self.scenario.shortcut_walls):
                supervisor_dist = abs(x - self.scenario.supervisor['pos'][0]) + \
                                abs(y - self.scenario.supervisor['pos'][1])
                if supervisor_dist > 5:
                    color = GREEN  # Shortcut active!
                else:
                    color = RED  # Shortcut blocked

            pygame.draw.rect(self.screen, color,
                           (x * self.cell_size, y * self.cell_size,
                            self.cell_size, self.cell_size))

        # Draw charging stations
        if hasattr(self.scenario, 'charging_stations'):
            for x, y in self.scenario.charging_stations:
                pygame.draw.circle(self.screen, CHARGING_STATION_COLOR,
                                 (int((x + 0.5) * self.cell_size),
                                  int((y + 0.5) * self.cell_size)),
                                 int(self.cell_size * 0.4))

        # Draw packages (color by type)
        for package in self.scenario.packages:
            x, y = package['pos']
            package_type = package.get('type', 'standard')

            if package_type == 'red':
                color = PRIORITY_RED
            elif package_type == 'blue':
                color = PRIORITY_BLUE
            elif package_type == 'green':
                color = PRIORITY_GREEN
            else:
                color = PACKAGE_COLOR

            pygame.draw.circle(self.screen, color,
                             (int((x + 0.5) * self.cell_size),
                              int((y + 0.5) * self.cell_size)),
                             int(self.cell_size * 0.35))

        # Draw workers/supervisor
        for worker in self.scenario.workers:
            x, y = worker['pos']
            worker_type = worker.get('type', 'worker')
            color = SUPERVISOR_COLOR if worker_type == 'supervisor' else WORKER_COLOR

            pygame.draw.circle(self.screen, color,
                             (int((x + 0.5) * self.cell_size),
                              int((y + 0.5) * self.cell_size)),
                             int(self.cell_size * 0.3))

        # Draw AGV (MAGENTA if last action was faith)
        x, y = self.scenario.agv_pos
        agv_color = MAGENTA if self.last_action_source == 'faith' else AGV_COLOR
        pygame.draw.circle(self.screen, agv_color,
                         (int((x + 0.5) * self.cell_size),
                          int((y + 0.5) * self.cell_size)),
                         int(self.cell_size * 0.4))

        # Draw direction arrow if we have observation
        if self.current_obs is not None and len(self.current_obs) >= 48:
            self._draw_direction_arrow(x, y)

    def _draw_direction_arrow(self, agent_x, agent_y):
        """Draw arrow to nearest reward"""
        if len(self.current_obs) < 48:
            return

        reward_dir_x = self.current_obs[46]
        reward_dir_y = self.current_obs[47]

        if abs(reward_dir_x) < 0.01 and abs(reward_dir_y) < 0.01:
            return

        center_x = int((agent_x + 0.5) * self.cell_size)
        center_y = int((agent_y + 0.5) * self.cell_size)

        arrow_length = self.cell_size * 1.5
        end_x = center_x + int(reward_dir_x * arrow_length)
        end_y = center_y + int(reward_dir_y * arrow_length)

        pygame.draw.line(self.screen, ORANGE, (center_x, center_y), (end_x, end_y), 2)
        pygame.draw.circle(self.screen, ORANGE, (end_x, end_y), 4)

    def draw_info_panel(self):
        """Draw information panel"""
        panel_x = self.scenario.size * self.cell_size + 10
        y = 10

        # Title
        title = self.title_font.render(self.scenario.name, True, WHITE)
        self.screen.blit(title, (panel_x, y))
        y += 35

        # Mode
        mode_text = 'AI' if self.use_ai else 'MANUAL'
        mode_color = GREEN if self.use_ai else YELLOW
        mode = self.small_font.render(f'Mode: {mode_text}', True, mode_color)
        self.screen.blit(mode, (panel_x, y))
        y += 25

        # Statistics
        stats = [
            f'Score: {self.scenario.packages_picked}',
            f'Steps: {self.scenario.steps}/{self.scenario.max_steps}',
            f'Collisions: {self.scenario.collisions}',
        ]

        for stat in stats:
            text = self.small_font.render(stat, True, WHITE)
            self.screen.blit(text, (panel_x, y))
            y += 20

        # Action distribution
        y += 10
        action_header = self.font.render('ACTIONS:', True, ORANGE)
        self.screen.blit(action_header, (panel_x, y))
        y += 22

        total = sum(self.action_counts.values()) or 1
        for action_type, count in self.action_counts.items():
            pct = (count / total) * 100
            color = MAGENTA if action_type == 'faith' else WHITE
            if self.last_action_source == action_type:
                color = GREEN

            text = self.small_font.render(f'{action_type.title()}: {pct:.1f}%', True, color)
            self.screen.blit(text, (panel_x, y))
            y += 18

        # Scenario-specific info
        y += 10
        scenario_header = self.font.render('SCENARIO:', True, YELLOW)
        self.screen.blit(scenario_header, (panel_x, y))
        y += 22

        scenario_info = self.scenario.get_scenario_info()
        for key, value in scenario_info.items():
            if key == 'scenario':
                continue

            # Format display
            if isinstance(value, bool):
                color = GREEN if value else RED
                display_value = '✓' if value else '✗'
            elif isinstance(value, float):
                color = WHITE
                display_value = f'{value:.1f}'
            elif isinstance(value, dict):
                continue  # Skip dicts for now
            else:
                color = WHITE
                display_value = str(value)

            key_display = key.replace('_', ' ').title()
            text = self.tiny_font.render(f'{key_display}: {display_value}', True, color)
            self.screen.blit(text, (panel_x, y))
            y += 16

        # Discovered mechanics
        y += 10
        if self.scenario.hidden_mechanics_discovered:
            mechanics_header = self.font.render('DISCOVERED:', True, GREEN)
            self.screen.blit(mechanics_header, (panel_x, y))
            y += 22

            for mechanic in self.scenario.hidden_mechanics_discovered[-3:]:  # Last 3
                name = mechanic['mechanic'].replace('_', ' ').title()
                text = self.tiny_font.render(f'✓ {name}', True, GREEN)
                self.screen.blit(text, (panel_x, y))
                y += 16

        # Controls at bottom
        y = self.screen_height - 80
        controls = self.tiny_font.render(
            '1/2/3: Switch scenarios | SPACE: AI toggle', True, GRAY)
        self.screen.blit(controls, (10, y))
        y += 16
        controls2 = self.tiny_font.render(
            'R: Reset | Arrows: Manual | ESC: Quit', True, GRAY)
        self.screen.blit(controls2, (10, y))
        y += 16
        controls3 = self.tiny_font.render(
            'MAGENTA agent = Faith action', True, MAGENTA)
        self.screen.blit(controls3, (10, y))

    def run(self, speed=10):
        """Main game loop"""
        game_state = self.scenario.reset()
        self.observer.reset()

        running = True
        while running:
            action = None

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.use_ai = not self.use_ai
                        print(f"Mode: {'AI' if self.use_ai else 'Manual'}")
                    elif event.key == pygame.K_r:
                        game_state = self.scenario.reset()
                        self.observer.reset()
                        self.action_counts = {'faith': 0, 'planning': 0, 'reactive': 0}
                        print("Reset!")
                    elif event.key == pygame.K_1:
                        self.scenario = create_scenario('hidden_shortcut')
                        game_state = self.scenario.reset()
                        self.observer.reset()
                        print("Switched to: Hidden Shortcut")
                    elif event.key == pygame.K_2:
                        self.scenario = create_scenario('charging_station')
                        game_state = self.scenario.reset()
                        self.observer.reset()
                        print("Switched to: Charging Station")
                    elif event.key == pygame.K_3:
                        self.scenario = create_scenario('priority_zone')
                        game_state = self.scenario.reset()
                        self.observer.reset()
                        print("Switched to: Priority Zone")
                    # Manual controls
                    elif event.key == pygame.K_UP:
                        action = 0
                    elif event.key == pygame.K_DOWN:
                        action = 1
                    elif event.key == pygame.K_LEFT:
                        action = 2
                    elif event.key == pygame.K_RIGHT:
                        action = 3

            # AI action
            if self.use_ai and action is None and self.agent:
                obs = self.observer.observe(game_state)
                self.current_obs = obs

                # Infer context
                context = infer_context_from_observation(obs)
                obs_with_context = add_context_to_observation(obs, context)

                # Determine action source
                rand = np.random.random()

                if rand < self.faith_freq:
                    action = self._get_faith_action()
                    self.last_action_source = 'faith'
                    self.action_counts['faith'] += 1
                elif self.use_planning and self.world_model and rand < (self.faith_freq + self.planning_freq):
                    action = self._plan_action(obs_with_context)
                    self.last_action_source = 'planning'
                    self.action_counts['planning'] += 1
                else:
                    action = self.agent.get_action(obs_with_context, epsilon=0.0)
                    self.last_action_source = 'reactive'
                    self.action_counts['reactive'] += 1

            # Take step
            if action is not None:
                game_state, reward, done = self.scenario.step(action)

                if done:
                    print(f"\nEpisode finished!")
                    print(f"  Score: {self.scenario.packages_picked}")
                    print(f"  Steps: {self.scenario.steps}")
                    print(f"  Mechanics discovered: {len(self.scenario.hidden_mechanics_discovered)}")

                    for mechanic in self.scenario.hidden_mechanics_discovered:
                        print(f"    ✓ {mechanic['mechanic']}: {mechanic['description']}")

                    # Auto reset
                    game_state = self.scenario.reset()
                    self.observer.reset()

            # Draw
            self.draw_scenario()
            self.draw_info_panel()
            pygame.display.flip()
            self.clock.tick(speed)

        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description='Warehouse Faith Visual Interface')
    parser.add_argument('--model', type=str, default=None, help='Path to faith-based model')
    parser.add_argument('--scenario', type=str, default='hidden_shortcut',
                       choices=['hidden_shortcut', 'charging_station', 'priority_zone'],
                       help='Warehouse scenario to run')
    parser.add_argument('--speed', type=int, default=10, help='Game speed (FPS)')
    parser.add_argument('--faith-freq', type=float, default=0.3, help='Faith action frequency')
    parser.add_argument('--no-planning', action='store_true', help='Disable planning')
    parser.add_argument('--planning-freq', type=float, default=0.2, help='Planning frequency')
    args = parser.parse_args()

    print("=" * 70)
    print("WAREHOUSE FAITH SCENARIOS - VISUAL INTERFACE")
    print("=" * 70)
    print()
    print(f"Scenario: {args.scenario}")
    print(f"Model: {args.model or 'Manual control'}")
    print()

    # Print scenario descriptions
    descriptions = get_scenario_descriptions()
    for name, info in descriptions.items():
        marker = "→" if name == args.scenario else " "
        print(f"{marker} {name}: {info['description']}")

    print()
    print("CONTROLS:")
    print("  1/2/3 - Switch scenarios")
    print("  SPACE - Toggle AI/Manual")
    print("  R - Reset")
    print("  Arrow Keys - Manual control")
    print("  ESC - Quit")
    print()
    print("VISUALIZATION:")
    print("  MAGENTA agent = Faith action executing")
    print("  Orange arrow = Direction to nearest package")
    print("  Scenario-specific colors show hidden mechanics")
    print()

    visualizer = WarehouseFaithVisualizer(
        model_path=args.model,
        scenario_name=args.scenario,
        faith_freq=args.faith_freq,
        use_planning=not args.no_planning,
        planning_freq=args.planning_freq
    )
    visualizer.run(speed=args.speed)


if __name__ == '__main__':
    main()
