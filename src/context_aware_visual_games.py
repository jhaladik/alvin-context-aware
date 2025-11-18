"""
Visual Test Games for Context-Aware Agent
Watch the agent play Snake, Pac-Man, and Dungeon with real-time context adaptation
"""
import pygame
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from planning_test_games import SnakeGame, PacManGame, DungeonGame
from context_aware_agent import (
    ContextAwareDQN,
    infer_context_from_observation,
    add_context_to_observation
)
from temporal_observer import TemporalFlowObserver

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
CYAN = (0, 255, 255)
GRAY = (128, 128, 128)
DARK_GREEN = (0, 100, 0)
GOLD = (255, 215, 0)
PURPLE = (128, 0, 128)


class ContextAwareVisualRunner:
    """Run games with context-aware agent and visual rendering"""

    def __init__(self, model_path, cell_size=25):
        self.cell_size = cell_size

        # Load context-aware agent
        self.agent = None
        self.observer = TemporalFlowObserver()
        self.current_context = None
        self.current_context_name = "Unknown"

        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            self.agent = ContextAwareDQN(obs_dim=95, action_dim=4)
            self.agent.load_state_dict(checkpoint['policy_net'])
            self.agent.eval()
            print(f"Loaded context-aware agent: {model_path}")
            print(f"  Episodes trained: {len(checkpoint.get('episode_rewards', []))}")
            print(f"  Input: 95-dim (92 temporal + 3 context)")
        else:
            print("No agent loaded - manual control only")

        # Initialize pygame
        pygame.init()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.clock = pygame.time.Clock()

        # Current game
        self.current_game = None
        self.game_name = ""
        self.game_state = None
        self.score = 0
        self.steps = 0
        self.use_ai = True if self.agent else False
        self.last_action = -1

        # Temporal info display
        self.current_obs = None
        self.reward_direction = (0, 0)
        self.danger_trend = 0
        self.progress_rate = 0

    def switch_game(self, game_type):
        """Switch to a different game"""
        if game_type == 'snake':
            self.current_game = SnakeGame(size=15)
            self.game_name = "SNAKE"
            self.grid_size = 15
        elif game_type == 'pacman':
            self.current_game = PacManGame(size=15)
            self.game_name = "PAC-MAN"
            self.grid_size = 15
        elif game_type == 'dungeon':
            self.current_game = DungeonGame(size=20)
            self.game_name = "DUNGEON"
            self.grid_size = 20

        self.game_state = self.current_game.reset()
        self.observer.reset()
        self.score = 0
        self.steps = 0
        self.last_action = -1
        self._update_temporal_info()

        # Resize window
        width = self.grid_size * self.cell_size + 220
        height = self.grid_size * self.cell_size + 80
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(f'Context-Aware Agent - {self.game_name}')

    def _update_temporal_info(self):
        """Extract temporal information from observation for display"""
        if self.current_obs is not None and len(self.current_obs) >= 92:
            # Reward direction (indices 46-47)
            self.reward_direction = (self.current_obs[46], self.current_obs[47])

            # Delta features start at 48
            # Danger trend at 48 + 24 + 6 + 8 + 2 = 88
            # Progress rate at 90
            if len(self.current_obs) > 90:
                self.danger_trend = self.current_obs[88]
                self.progress_rate = self.current_obs[90]

    def draw_snake(self):
        """Draw snake game"""
        game = self.current_game
        self.screen.fill(BLACK)

        # Draw boundary
        for i in range(self.grid_size):
            pygame.draw.rect(self.screen, DARK_GREEN,
                           (i * self.cell_size, 0, self.cell_size, self.cell_size))
            pygame.draw.rect(self.screen, DARK_GREEN,
                           (i * self.cell_size, (self.grid_size-1) * self.cell_size,
                            self.cell_size, self.cell_size))
            pygame.draw.rect(self.screen, DARK_GREEN,
                           (0, i * self.cell_size, self.cell_size, self.cell_size))
            pygame.draw.rect(self.screen, DARK_GREEN,
                           ((self.grid_size-1) * self.cell_size, i * self.cell_size,
                            self.cell_size, self.cell_size))

        # Draw snake body
        for x, y in game.snake[1:]:
            pygame.draw.rect(self.screen, GREEN,
                           (x * self.cell_size + 2, y * self.cell_size + 2,
                            self.cell_size - 4, self.cell_size - 4))

        # Draw snake head
        hx, hy = game.snake[0]
        pygame.draw.rect(self.screen, CYAN,
                       (hx * self.cell_size + 1, hy * self.cell_size + 1,
                        self.cell_size - 2, self.cell_size - 2))

        # Draw food
        fx, fy = game.food
        pygame.draw.circle(self.screen, RED,
                         (int((fx + 0.5) * self.cell_size),
                          int((fy + 0.5) * self.cell_size)),
                         int(self.cell_size * 0.4))

        # Draw direction arrow and detection rays
        self._draw_direction_arrow(hx, hy)

    def draw_pacman(self):
        """Draw Pac-Man game"""
        game = self.current_game
        self.screen.fill(BLACK)

        # Draw maze walls
        for x, y in game.walls:
            pygame.draw.rect(self.screen, BLUE,
                           (x * self.cell_size, y * self.cell_size,
                            self.cell_size, self.cell_size))

        # Draw pellets
        for px, py in game.pellets:
            pygame.draw.circle(self.screen, WHITE,
                             (int((px + 0.5) * self.cell_size),
                              int((py + 0.5) * self.cell_size)), 3)

        # Draw ghosts
        for ghost in game.ghosts:
            gx, gy = ghost['pos']
            pygame.draw.circle(self.screen, RED,
                             (int((gx + 0.5) * self.cell_size),
                              int((gy + 0.5) * self.cell_size)),
                             int(self.cell_size * 0.4))

        # Draw Pac-Man
        px, py = game.pacman_pos
        pygame.draw.circle(self.screen, YELLOW,
                         (int((px + 0.5) * self.cell_size),
                          int((py + 0.5) * self.cell_size)),
                         int(self.cell_size * 0.4))

        # Draw direction arrow and detection rays
        self._draw_direction_arrow(px, py)

    def draw_dungeon(self):
        """Draw dungeon game"""
        game = self.current_game
        self.screen.fill(BLACK)

        # Draw walls
        for x, y in game.walls:
            pygame.draw.rect(self.screen, GRAY,
                           (x * self.cell_size, y * self.cell_size,
                            self.cell_size, self.cell_size))

        # Draw treasure (single)
        tx, ty = game.treasure
        pygame.draw.circle(self.screen, GOLD,
                         (int((tx + 0.5) * self.cell_size),
                          int((ty + 0.5) * self.cell_size)), 5)

        # Draw enemies
        for enemy in game.enemies:
            mx, my = enemy['pos']
            pygame.draw.circle(self.screen, RED,
                             (int((mx + 0.5) * self.cell_size),
                              int((my + 0.5) * self.cell_size)),
                             int(self.cell_size * 0.35))

        # Draw player
        px, py = game.player_pos
        pygame.draw.circle(self.screen, GREEN,
                         (int((px + 0.5) * self.cell_size),
                          int((py + 0.5) * self.cell_size)),
                         int(self.cell_size * 0.4))

        # Draw direction arrow and detection rays
        self._draw_direction_arrow(px, py)

    def _draw_direction_arrow(self, agent_x, agent_y):
        """Draw arrow showing direction to nearest reward"""
        if abs(self.reward_direction[0]) < 0.01 and abs(self.reward_direction[1]) < 0.01:
            return

        # Agent center position
        center_x = int((agent_x + 0.5) * self.cell_size)
        center_y = int((agent_y + 0.5) * self.cell_size)

        # Arrow direction (scaled)
        arrow_length = self.cell_size * 2
        end_x = center_x + int(self.reward_direction[0] * arrow_length)
        end_y = center_y + int(self.reward_direction[1] * arrow_length)

        # Draw arrow line
        pygame.draw.line(self.screen, PURPLE, (center_x, center_y), (end_x, end_y), 3)

        # Draw arrowhead
        pygame.draw.circle(self.screen, PURPLE, (end_x, end_y), 5)

        # Draw entity and wall detection rays
        self._draw_detection_rays(center_x, center_y)

    def _draw_detection_rays(self, center_x, center_y):
        """Draw rays showing entity and wall detection in 8 directions"""
        if self.current_obs is None or len(self.current_obs) < 48:
            return

        # Ray directions: N, NE, E, SE, S, SW, W, NW
        ray_dirs = [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1)
        ]

        max_ray_length = self.cell_size * 4  # Visual max length

        for i, (dx, dy) in enumerate(ray_dirs):
            # Extract distances from state
            # Ray data: [reward_dist, entity_dist, wall_dist] for each of 8 rays
            base_idx = i * 3
            reward_dist = self.current_obs[base_idx]      # 0-1, higher = farther
            entity_dist = self.current_obs[base_idx + 1]  # 0-1, higher = farther
            wall_dist = self.current_obs[base_idx + 2]    # 0-1, higher = farther

            # Normalize direction
            length = np.sqrt(dx*dx + dy*dy)
            norm_dx = dx / length
            norm_dy = dy / length

            # Draw WALL ray (gray, thin)
            wall_ray_len = wall_dist * max_ray_length
            wall_end_x = center_x + int(norm_dx * wall_ray_len)
            wall_end_y = center_y + int(norm_dy * wall_ray_len)
            pygame.draw.line(self.screen, GRAY, (center_x, center_y),
                           (wall_end_x, wall_end_y), 1)

            # Draw ENTITY ray (orange/red based on distance)
            if entity_dist < 0.9:  # Only draw if entity detected
                entity_ray_len = entity_dist * max_ray_length
                # Color: red if close, orange if medium, light orange if far
                if entity_dist < 0.3:
                    entity_color = RED
                elif entity_dist < 0.6:
                    entity_color = ORANGE
                else:
                    entity_color = (255, 200, 100)  # Light orange

                entity_end_x = center_x + int(norm_dx * entity_ray_len)
                entity_end_y = center_y + int(norm_dy * entity_ray_len)
                pygame.draw.line(self.screen, entity_color, (center_x, center_y),
                               (entity_end_x, entity_end_y), 2)
                # Small circle at end to show distance
                pygame.draw.circle(self.screen, entity_color, (entity_end_x, entity_end_y), 3)

    def draw_info_panel(self):
        """Draw info panel on the right"""
        panel_x = self.grid_size * self.cell_size + 10
        y = 10

        # Game name
        title = self.font.render(self.game_name, True, WHITE)
        self.screen.blit(title, (panel_x, y))
        y += 30

        # Mode
        mode_text = 'AI' if self.use_ai else 'MANUAL'
        mode_color = GREEN if self.use_ai else YELLOW
        mode = self.font.render(f'Mode: {mode_text}', True, mode_color)
        self.screen.blit(mode, (panel_x, y))
        y += 30

        # Stats
        stats = [
            f'Score: {self.score}',
            f'Steps: {self.steps}',
        ]

        if hasattr(self.current_game, 'lives'):
            stats.append(f'Lives: {self.current_game.lives}')

        if self.game_name == 'SNAKE':
            stats.append(f'Length: {len(self.current_game.snake)}')
        elif self.game_name == 'PAC-MAN':
            stats.append(f'Pellets: {len(self.current_game.pellets)}')
        elif self.game_name == 'DUNGEON':
            # Only 1 treasure in DungeonGame
            treasure_collected = 1 if self.score >= 10 else 0
            stats.append(f'Treasure: {treasure_collected}/1')

        for stat in stats:
            text = self.small_font.render(stat, True, WHITE)
            self.screen.blit(text, (panel_x, y))
            y += 20

        # Context information (KEY ADDITION!)
        y += 10
        context_title = self.font.render('CONTEXT:', True, WHITE)
        self.screen.blit(context_title, (panel_x, y))
        y += 25

        if self.current_context_name == "SNAKE":
            context_color = GREEN
            context_desc = "No entities"
        elif self.current_context_name == "BALANCED":
            context_color = ORANGE
            context_desc = "2-3 entities"
        else:
            context_color = RED
            context_desc = "4+ entities"

        context_text = self.small_font.render(self.current_context_name, True, context_color)
        self.screen.blit(context_text, (panel_x, y))
        y += 18

        desc_text = self.small_font.render(context_desc, True, GRAY)
        self.screen.blit(desc_text, (panel_x, y))
        y += 30

        # TEMPORAL INFO
        temporal_header = self.font.render('TEMPORAL', True, PURPLE)
        self.screen.blit(temporal_header, (panel_x, y))
        y += 25

        # Reward direction
        dir_text = f'Target: ({self.reward_direction[0]:.2f}, {self.reward_direction[1]:.2f})'
        dir_render = self.small_font.render(dir_text, True, CYAN)
        self.screen.blit(dir_render, (panel_x, y))
        y += 20

        # Danger trend
        danger_color = RED if self.danger_trend > 0.1 else GREEN
        danger_text = f'Danger: {self.danger_trend:.3f}'
        danger_render = self.small_font.render(danger_text, True, danger_color)
        self.screen.blit(danger_render, (panel_x, y))
        y += 20

        # Progress rate
        progress_color = GREEN if self.progress_rate > 0 else RED
        progress_text = f'Progress: {self.progress_rate:.3f}'
        progress_render = self.small_font.render(progress_text, True, progress_color)
        self.screen.blit(progress_render, (panel_x, y))
        y += 20

        # Controls
        y = self.grid_size * self.cell_size + 10
        controls = self.small_font.render(
            '1:Snake 2:PacMan 3:Dungeon', True, GRAY)
        self.screen.blit(controls, (10, y))
        y += 18
        controls2 = self.small_font.render(
            'SPACE:AI R:Reset ESC:Quit', True, GRAY)
        self.screen.blit(controls2, (10, y))

    def draw(self):
        """Draw current game"""
        if self.game_name == 'SNAKE':
            self.draw_snake()
        elif self.game_name == 'PAC-MAN':
            self.draw_pacman()
        elif self.game_name == 'DUNGEON':
            self.draw_dungeon()

        self.draw_info_panel()
        pygame.display.flip()

    def run(self, speed=10):
        """Main game loop"""
        # Start with snake
        self.switch_game('snake')

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
                        self.game_state = self.current_game.reset()
                        self.observer.reset()
                        self.score = 0
                        self.steps = 0
                        self._update_temporal_info()
                        print("Reset!")
                    elif event.key == pygame.K_1:
                        self.switch_game('snake')
                        print("Switched to SNAKE")
                    elif event.key == pygame.K_2:
                        self.switch_game('pacman')
                        print("Switched to PAC-MAN")
                    elif event.key == pygame.K_3:
                        self.switch_game('dungeon')
                        print("Switched to DUNGEON")
                    # Manual controls
                    elif event.key == pygame.K_UP:
                        action = 0
                    elif event.key == pygame.K_DOWN:
                        action = 1
                    elif event.key == pygame.K_LEFT:
                        action = 2
                    elif event.key == pygame.K_RIGHT:
                        action = 3

            # AI action with context awareness
            if self.use_ai and action is None and self.agent:
                # Convert game_state to observation
                obs = self.observer.observe(self.game_state)
                self.current_obs = obs

                # Infer context from observation
                self.current_context = infer_context_from_observation(obs)

                # Determine context name
                if self.current_context[0] == 1.0:
                    self.current_context_name = "SNAKE"
                elif self.current_context[1] == 1.0:
                    self.current_context_name = "BALANCED"
                else:
                    self.current_context_name = "SURVIVAL"

                # Add context to observation
                obs_with_context = add_context_to_observation(obs, self.current_context)

                # Get action
                action = self.agent.get_action(obs_with_context, epsilon=0.0)

                # Update temporal info for display
                self._update_temporal_info()

            # Take step
            if action is not None:
                self.game_state, reward, done = self.current_game.step(action)
                self.score = self.game_state.get('score', 0)
                self.steps = self.current_game.steps
                self.last_action = action

                if done:
                    print(f"{self.game_name} finished! Score: {self.score} | Context: {self.current_context_name}")
                    # Auto reset
                    self.game_state = self.current_game.reset()
                    self.observer.reset()
                    self.score = 0
                    self.steps = 0

            # Draw
            self.draw()
            self.clock.tick(speed)

        pygame.quit()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Context-Aware Visual Test Games')
    parser.add_argument('--model', type=str, default=None, help='Path to context-aware agent model')
    parser.add_argument('--speed', type=int, default=10, help='Game speed (FPS)')
    args = parser.parse_args()

    print("=" * 60)
    print("CONTEXT-AWARE VISUAL GAME TESTER")
    print("Test Context-Aware Agent on Snake, Pac-Man, Dungeon")
    print("=" * 60)
    print()
    print("CONTROLS:")
    print("  1 - Switch to Snake")
    print("  2 - Switch to Pac-Man")
    print("  3 - Switch to Dungeon")
    print("  SPACE - Toggle AI/Manual mode")
    print("  R - Reset current game")
    print("  Arrow Keys - Manual control")
    print("  ESC - Quit")
    print()
    print("CONTEXT MODES:")
    print("  GREEN (SNAKE): No entities detected - Pure collection")
    print("  ORANGE (BALANCED): 2-3 entities - Tactical gameplay")
    print("  RED (SURVIVAL): 4+ entities - High threat avoidance")
    print()
    print("VISUAL FEATURES:")
    print("  Purple arrow: Direction to nearest reward")
    print("  Gray rays: Wall detection in 8 directions")
    print("  Orange/Red rays: Entity detection (red=close, orange=far)")
    print("  Detection rays show what agent 'sees'")
    print()

    runner = ContextAwareVisualRunner(model_path=args.model)
    runner.run(speed=args.speed)


if __name__ == '__main__':
    main()
