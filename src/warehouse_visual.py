"""
Warehouse AGV Simulation - Context-Aware Robot Navigation

Realistic warehouse operations simulation using the context-aware agent.
The agent controls an AGV (Automated Guided Vehicle) that must:
- Pick packages from designated locations
- Avoid collisions with workers and other robots
- Navigate efficiently through warehouse aisles
- Adapt behavior based on warehouse traffic density

Mapping to trained model:
- Entities = Workers, other robots (dynamic obstacles)
- Rewards = Packages to pick
- Walls = Shelves, racks, warehouse boundaries
- Context automatically adapts to traffic density
"""

import pygame
import sys
import os
import torch
import numpy as np
import random

sys.path.insert(0, os.path.dirname(__file__))

from context_aware_agent import (
    ContextAwareDQN,
    infer_context_from_observation,
    add_context_to_observation
)
from core.temporal_observer import TemporalFlowObserver

# Colors - Simple and clean
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)
DARK_GRAY = (100, 100, 100)
LIGHT_GRAY = (220, 220, 220)

# Warehouse theme
FLOOR_COLOR = (240, 240, 240)
SHELF_COLOR = (139, 90, 60)  # Brown
AGV_COLOR = (30, 144, 255)    # Blue
WORKER_COLOR = (220, 50, 50)  # Red
PACKAGE_COLOR = (255, 215, 0) # Gold
AISLE_COLOR = (200, 200, 200)

# Status colors
GREEN = (50, 200, 50)
YELLOW = (255, 200, 0)
RED = (220, 50, 50)
ORANGE = (255, 140, 0)


class WarehouseGame:
    """
    Warehouse AGV Navigation Game

    Difficulty levels based on warehouse traffic:
    - Easy (0 workers): Off-hours, empty warehouse
    - Medium (2-3 workers): Normal operations
    - Hard (4-6 workers): Peak hours, busy warehouse
    """

    def __init__(self, size=20, difficulty='medium'):
        self.size = size
        self.difficulty = difficulty
        self.steps = 0
        self.max_steps = 500

        # Warehouse layout - realistic aisle structure
        self.walls = set()
        self.aisles = []
        self._create_warehouse_layout()

        # Difficulty determines number of workers/obstacles
        self.difficulty_config = {
            'easy': (0, 0),      # No workers
            'medium': (2, 3),    # 2-3 workers
            'hard': (4, 6)       # 4-6 workers
        }

        # Statistics tracking
        self.packages_picked = 0
        self.total_packages_spawned = 0
        self.collisions = 0
        self.total_distance = 0
        self.idle_time = 0
        self.context_time = {'easy': 0, 'medium': 0, 'hard': 0}

        self.reset()

    def _create_warehouse_layout(self):
        """Create realistic warehouse with aisles and shelving"""
        # Outer walls
        for i in range(self.size):
            self.walls.add((i, 0))
            self.walls.add((i, self.size - 1))
            self.walls.add((0, i))
            self.walls.add((self.size - 1, i))

        # Create 3 vertical aisles with shelves
        # Aisle structure: shelf | aisle | shelf | aisle | shelf | aisle | shelf
        aisle_positions = [5, 10, 15]

        for aisle_x in aisle_positions:
            self.aisles.append(aisle_x)
            # Add shelves on both sides of aisle (with gaps for access)
            for y in range(2, self.size - 2):
                # Add shelves, but leave gaps every 5 units for cross-aisles
                if y % 5 != 0:
                    if aisle_x > 2:
                        self.walls.add((aisle_x - 1, y))  # Left shelf
                    if aisle_x < self.size - 3:
                        self.walls.add((aisle_x + 1, y))  # Right shelf

    def reset(self):
        """Reset warehouse to initial state"""
        self.steps = 0
        self.done = False

        # Place AGV in loading dock area (bottom left)
        self.agv_pos = (2, self.size - 3)
        self.last_pos = self.agv_pos

        # Initialize empty lists first
        self.packages = []
        self.workers = []

        # Spawn workers based on difficulty
        min_workers, max_workers = self.difficulty_config[self.difficulty]
        num_workers = random.randint(min_workers, max_workers)
        for _ in range(num_workers):
            self._spawn_worker()

        # Spawn initial packages (3-5 packages)
        num_packages = random.randint(3, 5)
        for _ in range(num_packages):
            self._spawn_package()

        # Current task
        self.current_target = None
        if self.packages:
            self.current_target = self.packages[0]

        return self._get_state()

    def _spawn_package(self):
        """Spawn package at random shelf location"""
        attempts = 0
        while attempts < 100:
            # Packages spawn near shelves (in picking locations)
            x = random.choice([4, 6, 9, 11, 14, 16])
            y = random.randint(3, self.size - 4)
            pos = (x, y)

            if (pos not in self.walls and
                pos != self.agv_pos and
                pos not in self.packages and
                pos not in [w['pos'] for w in self.workers]):
                self.packages.append(pos)
                self.total_packages_spawned += 1
                return pos
            attempts += 1
        return None

    def _spawn_worker(self):
        """Spawn worker with random patrol pattern"""
        attempts = 0
        while attempts < 100:
            x = random.randint(2, self.size - 3)
            y = random.randint(2, self.size - 3)
            pos = (x, y)

            if (pos not in self.walls and
                pos != self.agv_pos and
                pos not in self.packages):

                # Worker has velocity (walking speed)
                dx = random.choice([-1, 0, 1])
                dy = random.choice([-1, 0, 1])

                self.workers.append({
                    'pos': pos,
                    'velocity': (dx, dy),
                    'idle_counter': 0
                })
                return
            attempts += 1

    def _update_workers(self):
        """Update worker positions (realistic movement patterns)"""
        for worker in self.workers:
            # Workers occasionally change direction or pause
            if random.random() < 0.1:  # 10% chance to change behavior
                if random.random() < 0.3:  # Pause
                    worker['velocity'] = (0, 0)
                    worker['idle_counter'] = random.randint(5, 15)
                else:  # Change direction
                    worker['velocity'] = (
                        random.choice([-1, 0, 1]),
                        random.choice([-1, 0, 1])
                    )

            # Handle idle time
            if worker['idle_counter'] > 0:
                worker['idle_counter'] -= 1
                continue

            # Move worker
            dx, dy = worker['velocity']
            new_x = worker['pos'][0] + dx
            new_y = worker['pos'][1] + dy
            new_pos = (new_x, new_y)

            # Check if new position is valid
            if (new_pos not in self.walls and
                0 < new_x < self.size - 1 and
                0 < new_y < self.size - 1):
                worker['pos'] = new_pos
            else:
                # Hit obstacle, change direction
                worker['velocity'] = (
                    random.choice([-1, 0, 1]),
                    random.choice([-1, 0, 1])
                )

    def step(self, action):
        """Execute action: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT"""
        if self.done:
            return self._get_state(), 0, True

        self.steps += 1

        # Move AGV
        moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
        dx, dy = moves[action]
        new_x = self.agv_pos[0] + dx
        new_y = self.agv_pos[1] + dy
        new_pos = (new_x, new_y)

        reward = -0.1  # Small time penalty for efficiency

        # Check collision with walls
        if new_pos in self.walls:
            reward = -2.0  # Penalty for hitting shelf
            self.collisions += 1
        else:
            # Valid move
            self.last_pos = self.agv_pos
            self.agv_pos = new_pos

            # Calculate distance traveled
            self.total_distance += 1

            # Check if picked up package
            if self.agv_pos in self.packages:
                self.packages.remove(self.agv_pos)
                self.packages_picked += 1
                reward = 10.0  # Good reward for picking package

                # Spawn new package to keep warehouse active
                self._spawn_package()

                # Update target
                if self.packages:
                    self.current_target = self.packages[0]

        # Update workers
        self._update_workers()

        # Check collision with workers
        for worker in self.workers:
            if self.agv_pos == worker['pos']:
                reward = -5.0  # Penalty for worker collision (safety critical!)
                self.collisions += 1
                break

        # Check if episode done
        if self.steps >= self.max_steps:
            self.done = True

        return self._get_state(), reward, self.done

    def _get_state(self):
        """Get warehouse state for observer"""
        # Build entity list (workers as dynamic obstacles)
        entities = []
        for worker in self.workers:
            entities.append({
                'pos': worker['pos'],
                'velocity': worker['velocity'],
                'danger': 1.0  # Workers are obstacles to avoid
            })

        return {
            'agent_pos': self.agv_pos,
            'walls': self.walls,
            'rewards': self.packages.copy(),
            'entities': entities,
            'grid_size': (self.size, self.size),
            'score': self.packages_picked,
            'done': self.done
        }

    def get_statistics(self):
        """Get comprehensive operational statistics"""
        efficiency = 0.0
        if self.total_distance > 0:
            efficiency = (self.packages_picked / self.total_distance) * 100

        collision_rate = 0.0
        if self.steps > 0:
            collision_rate = (self.collisions / self.steps) * 100

        return {
            'packages_picked': self.packages_picked,
            'packages_available': len(self.packages),
            'total_spawned': self.total_packages_spawned,
            'collisions': self.collisions,
            'collision_rate': collision_rate,
            'distance': self.total_distance,
            'efficiency': efficiency,
            'steps': self.steps,
            'workers': len(self.workers)
        }


class WarehouseVisualizer:
    """Visual renderer for warehouse simulation"""

    def __init__(self, model_path=None, difficulty='medium'):
        pygame.init()

        # Window setup
        self.cell_size = 30
        self.grid_size = 20
        self.info_panel_width = 350
        self.screen_width = self.grid_size * self.cell_size + self.info_panel_width
        self.screen_height = self.grid_size * self.cell_size

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Warehouse AGV - Context-Aware Navigation")

        # Fonts
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)
        self.title_font = pygame.font.Font(None, 28)

        # Game
        self.difficulty = difficulty
        self.game = WarehouseGame(size=self.grid_size, difficulty=difficulty)
        self.observer = TemporalFlowObserver()

        # Agent
        self.agent = None
        self.use_ai = True

        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            self.agent = ContextAwareDQN(obs_dim=95, action_dim=4)
            self.agent.load_state_dict(checkpoint['policy_net'])
            self.agent.eval()
            print(f"Loaded model: {model_path}")

        # State
        self.game_state = self.game.reset()
        self.observer.reset()
        self.current_obs = None
        self.current_context = None
        self.current_context_name = "Unknown"

    def draw_warehouse(self):
        """Draw warehouse with simple, clear graphics"""
        # Background - floor
        self.screen.fill(FLOOR_COLOR)

        # Draw grid lines (subtle)
        for i in range(self.grid_size + 1):
            pygame.draw.line(self.screen, LIGHT_GRAY,
                           (i * self.cell_size, 0),
                           (i * self.cell_size, self.screen_height), 1)
            pygame.draw.line(self.screen, LIGHT_GRAY,
                           (0, i * self.cell_size),
                           (self.grid_size * self.cell_size, i * self.cell_size), 1)

        # Draw shelves (walls)
        for x, y in self.game.walls:
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                             self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, SHELF_COLOR, rect)
            pygame.draw.rect(self.screen, DARK_GRAY, rect, 1)

        # Draw packages
        for px, py in self.game.packages:
            center_x = px * self.cell_size + self.cell_size // 2
            center_y = py * self.cell_size + self.cell_size // 2
            # Draw as simple box
            size = self.cell_size // 3
            rect = pygame.Rect(center_x - size//2, center_y - size//2, size, size)
            pygame.draw.rect(self.screen, PACKAGE_COLOR, rect)
            pygame.draw.rect(self.screen, ORANGE, rect, 2)

        # Draw workers
        for worker in self.game.workers:
            wx, wy = worker['pos']
            center_x = wx * self.cell_size + self.cell_size // 2
            center_y = wy * self.cell_size + self.cell_size // 2
            # Draw as circle
            pygame.draw.circle(self.screen, WORKER_COLOR,
                             (center_x, center_y), self.cell_size // 3)
            # Draw velocity indicator (small arrow)
            vx, vy = worker['velocity']
            if vx != 0 or vy != 0:
                arrow_end = (center_x + vx * 10, center_y + vy * 10)
                pygame.draw.line(self.screen, WHITE,
                               (center_x, center_y), arrow_end, 2)

        # Draw AGV
        agv_x, agv_y = self.game.agv_pos
        center_x = agv_x * self.cell_size + self.cell_size // 2
        center_y = agv_y * self.cell_size + self.cell_size // 2
        # Draw as rounded square
        size = self.cell_size // 2
        rect = pygame.Rect(center_x - size//2, center_y - size//2, size, size)
        pygame.draw.rect(self.screen, AGV_COLOR, rect, border_radius=5)
        pygame.draw.rect(self.screen, WHITE, rect, 2, border_radius=5)

    def draw_statistics(self):
        """Draw comprehensive statistics panel"""
        panel_x = self.grid_size * self.cell_size
        y_offset = 10

        # Background
        panel_rect = pygame.Rect(panel_x, 0, self.info_panel_width, self.screen_height)
        pygame.draw.rect(self.screen, WHITE, panel_rect)
        pygame.draw.line(self.screen, DARK_GRAY, (panel_x, 0), (panel_x, self.screen_height), 2)

        # Title
        title = self.title_font.render("WAREHOUSE OPS", True, BLACK)
        self.screen.blit(title, (panel_x + 10, y_offset))
        y_offset += 35

        # Difficulty badge
        diff_color = GREEN if self.difficulty == 'easy' else (YELLOW if self.difficulty == 'medium' else RED)
        diff_text = f"Mode: {self.difficulty.upper()}"
        diff_surf = self.small_font.render(diff_text, True, WHITE)
        diff_rect = pygame.Rect(panel_x + 10, y_offset, 120, 25)
        pygame.draw.rect(self.screen, diff_color, diff_rect, border_radius=3)
        self.screen.blit(diff_surf, (panel_x + 20, y_offset + 5))
        y_offset += 35

        # Context indicator
        ctx_colors = {'EASY': GREEN, 'MEDIUM': YELLOW, 'HARD': RED}
        ctx_color = ctx_colors.get(self.current_context_name, GRAY)
        ctx_text = f"Context: {self.current_context_name}"
        ctx_surf = self.small_font.render(ctx_text, True, WHITE)
        ctx_rect = pygame.Rect(panel_x + 10, y_offset, 150, 25)
        pygame.draw.rect(self.screen, ctx_color, ctx_rect, border_radius=3)
        self.screen.blit(ctx_surf, (panel_x + 20, y_offset + 5))
        y_offset += 40

        # Get statistics
        stats = self.game.get_statistics()

        # Performance metrics
        self._draw_stat_line(panel_x, y_offset, "PERFORMANCE", None, True)
        y_offset += 25

        self._draw_stat_line(panel_x, y_offset, "Packages Picked",
                            f"{stats['packages_picked']}", False, GREEN)
        y_offset += 22

        self._draw_stat_line(panel_x, y_offset, "Available",
                            f"{stats['packages_available']}", False)
        y_offset += 22

        eff_color = GREEN if stats['efficiency'] > 5 else (YELLOW if stats['efficiency'] > 2 else RED)
        self._draw_stat_line(panel_x, y_offset, "Efficiency",
                            f"{stats['efficiency']:.1f}%", False, eff_color)
        y_offset += 30

        # Safety metrics
        self._draw_stat_line(panel_x, y_offset, "SAFETY", None, True)
        y_offset += 25

        col_color = GREEN if stats['collisions'] == 0 else (YELLOW if stats['collisions'] < 3 else RED)
        self._draw_stat_line(panel_x, y_offset, "Collisions",
                            f"{stats['collisions']}", False, col_color)
        y_offset += 22

        rate_color = GREEN if stats['collision_rate'] < 1 else (YELLOW if stats['collision_rate'] < 3 else RED)
        self._draw_stat_line(panel_x, y_offset, "Collision Rate",
                            f"{stats['collision_rate']:.1f}%", False, rate_color)
        y_offset += 30

        # Operational metrics
        self._draw_stat_line(panel_x, y_offset, "OPERATIONS", None, True)
        y_offset += 25

        self._draw_stat_line(panel_x, y_offset, "Workers Active",
                            f"{stats['workers']}", False)
        y_offset += 22

        self._draw_stat_line(panel_x, y_offset, "Distance",
                            f"{stats['distance']}", False)
        y_offset += 22

        self._draw_stat_line(panel_x, y_offset, "Steps",
                            f"{stats['steps']}/{self.game.max_steps}", False)
        y_offset += 35

        # Progress bar
        progress = stats['steps'] / self.game.max_steps
        self._draw_progress_bar(panel_x + 10, y_offset, 330, 20, progress)
        y_offset += 35

        # Controls
        self._draw_stat_line(panel_x, y_offset, "CONTROLS", None, True)
        y_offset += 25

        controls = [
            "SPACE: AI Toggle",
            "R: Reset",
            "1/2/3: Difficulty",
            "ESC: Quit"
        ]
        for ctrl in controls:
            ctrl_surf = self.small_font.render(ctrl, True, DARK_GRAY)
            self.screen.blit(ctrl_surf, (panel_x + 15, y_offset))
            y_offset += 20

    def _draw_stat_line(self, panel_x, y, label, value, is_header, value_color=None):
        """Helper to draw a statistics line"""
        if is_header:
            text = self.font.render(label, True, BLACK)
            self.screen.blit(text, (panel_x + 10, y))
            # Underline
            pygame.draw.line(self.screen, DARK_GRAY,
                           (panel_x + 10, y + 22),
                           (panel_x + 340, y + 22), 1)
        else:
            label_surf = self.small_font.render(label, True, DARK_GRAY)
            self.screen.blit(label_surf, (panel_x + 15, y))

            if value:
                color = value_color if value_color else BLACK
                value_surf = self.small_font.render(value, True, color)
                value_rect = value_surf.get_rect(right=panel_x + 340, centery=y + 10)
                self.screen.blit(value_surf, value_rect)

    def _draw_progress_bar(self, x, y, width, height, progress):
        """Draw progress bar"""
        # Background
        pygame.draw.rect(self.screen, LIGHT_GRAY, (x, y, width, height), border_radius=3)
        # Progress
        if progress > 0:
            prog_width = int(width * progress)
            color = GREEN if progress < 0.5 else (YELLOW if progress < 0.8 else RED)
            pygame.draw.rect(self.screen, color, (x, y, prog_width, height), border_radius=3)
        # Border
        pygame.draw.rect(self.screen, DARK_GRAY, (x, y, width, height), 2, border_radius=3)

    def run(self, speed=10):
        """Main game loop"""
        clock = pygame.time.Clock()
        running = True

        while running:
            # Handle events
            action = None
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
                        self.game_state = self.game.reset()
                        self.observer.reset()
                        print("Reset!")
                    elif event.key == pygame.K_1:
                        self.difficulty = 'easy'
                        self.game = WarehouseGame(size=self.grid_size, difficulty='easy')
                        self.game_state = self.game.reset()
                        self.observer.reset()
                        print("Difficulty: EASY")
                    elif event.key == pygame.K_2:
                        self.difficulty = 'medium'
                        self.game = WarehouseGame(size=self.grid_size, difficulty='medium')
                        self.game_state = self.game.reset()
                        self.observer.reset()
                        print("Difficulty: MEDIUM")
                    elif event.key == pygame.K_3:
                        self.difficulty = 'hard'
                        self.game = WarehouseGame(size=self.grid_size, difficulty='hard')
                        self.game_state = self.game.reset()
                        self.observer.reset()
                        print("Difficulty: HARD")
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
                obs = self.observer.observe(self.game_state)
                self.current_obs = obs

                # Infer context
                self.current_context = infer_context_from_observation(obs)

                # Map context to warehouse traffic levels
                if self.current_context[0] == 1.0:
                    self.current_context_name = "EASY"  # No obstacles
                elif self.current_context[1] == 1.0:
                    self.current_context_name = "MEDIUM"  # Some obstacles
                else:
                    self.current_context_name = "HARD"  # Many obstacles

                obs_with_context = add_context_to_observation(obs, self.current_context)
                action = self.agent.get_action(obs_with_context, epsilon=0.0)

            # Take step
            if action is not None:
                self.game_state, reward, done = self.game.step(action)

                if done:
                    stats = self.game.get_statistics()
                    print(f"Shift complete!")
                    print(f"  Packages picked: {stats['packages_picked']}")
                    print(f"  Efficiency: {stats['efficiency']:.1f}%")
                    print(f"  Collisions: {stats['collisions']}")
                    # Auto reset
                    self.game_state = self.game.reset()
                    self.observer.reset()

            # Draw
            self.draw_warehouse()
            self.draw_statistics()
            pygame.display.flip()

            clock.tick(speed)

        pygame.quit()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Warehouse AGV Simulation')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model')
    parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard'],
                       default='medium', help='Warehouse traffic level')
    parser.add_argument('--speed', type=int, default=10, help='Simulation speed (FPS)')

    args = parser.parse_args()

    print("=" * 60)
    print("WAREHOUSE AGV SIMULATION")
    print("=" * 60)
    print(f"Difficulty: {args.difficulty.upper()}")
    if args.model:
        print(f"Model: {args.model}")
    print()
    print("Starting simulation...")
    print()

    viz = WarehouseVisualizer(model_path=args.model, difficulty=args.difficulty)
    viz.run(speed=args.speed)


if __name__ == '__main__':
    main()
