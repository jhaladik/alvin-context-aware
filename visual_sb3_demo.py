"""
Visual Demo: Stable-Baselines3 Snake Agent
Shows the trained PPO model playing Snake with pygame visualization
"""
import pygame
import numpy as np
import sys
import os
from stable_baselines3 import PPO

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from core.planning_test_games import SnakeGame

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
DARK_GREEN = (0, 100, 0)


class VisualSnakeDemo:
    """Visual demo of trained SB3 agent playing Snake"""

    def __init__(self, model_path, cell_size=30, fps=10):
        pygame.init()

        self.cell_size = cell_size
        self.fps = fps
        self.game = SnakeGame(size=20, num_pellets=10)
        self.size = 20

        # Load trained model
        print(f"Loading model from {model_path}...")
        try:
            self.model = PPO.load(model_path)
            print("Model loaded successfully!")

            # Print model info
            print(f"\nModel Info:")
            print(f"  Learning rate: {self.model.learning_rate}")
            print(f"  Batch size: {self.model.batch_size}")
            print(f"  Gamma: {self.model.gamma}")
            print()
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

        # Setup display
        self.screen_width = self.size * cell_size
        self.screen_height = self.size * cell_size + 100  # Extra space for stats
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("SB3 Snake Agent Demo")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # Stats
        self.episode = 0
        self.total_episodes = 0
        self.total_score = 0
        self.best_score = 0
        self.current_score = 0
        self.current_steps = 0

    def _get_obs(self, state):
        """Convert game state to observation (flattened grid)"""
        grid = np.zeros((self.size, self.size), dtype=np.float32)

        # Add food
        for fx, fy in state.get('food_positions', []):
            if 0 <= fx < self.size and 0 <= fy < self.size:
                grid[fy, fx] = 1.0

        # Add snake body
        for bx, by in state.get('snake_body', []):
            if 0 <= bx < self.size and 0 <= by < self.size:
                grid[by, bx] = 2.0

        # Add agent/head
        agent_x, agent_y = state.get('agent_pos', (self.size//2, self.size//2))
        if 0 <= agent_x < self.size and 0 <= agent_y < self.size:
            grid[agent_y, agent_x] = 3.0

        return grid.flatten()

    def render_game(self, state):
        """Render the game state with pygame"""
        self.screen.fill(BLACK)

        # Draw grid
        for x in range(self.size):
            for y in range(self.size):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, GRAY, rect, 1)

        # Draw food
        for fx, fy in state.get('food_positions', []):
            if 0 <= fx < self.size and 0 <= fy < self.size:
                center = (
                    int((fx + 0.5) * self.cell_size),
                    int((fy + 0.5) * self.cell_size)
                )
                pygame.draw.circle(self.screen, GREEN, center, self.cell_size // 3)

        # Draw snake body
        for bx, by in state.get('snake_body', []):
            if 0 <= bx < self.size and 0 <= by < self.size:
                rect = pygame.Rect(
                    bx * self.cell_size + 3,
                    by * self.cell_size + 3,
                    self.cell_size - 6,
                    self.cell_size - 6
                )
                pygame.draw.rect(self.screen, DARK_GREEN, rect)

        # Draw agent head (yellow)
        agent_x, agent_y = state.get('agent_pos', (self.size//2, self.size//2))
        if 0 <= agent_x < self.size and 0 <= agent_y < self.size:
            center = (
                int((agent_x + 0.5) * self.cell_size),
                int((agent_y + 0.5) * self.cell_size)
            )
            pygame.draw.circle(self.screen, YELLOW, center, self.cell_size // 2 - 2)

        # Draw stats panel
        stats_y = self.size * self.cell_size + 10

        # Current stats
        score_text = self.font.render(f"Score: {self.current_score}", True, WHITE)
        self.screen.blit(score_text, (10, stats_y))

        steps_text = self.small_font.render(f"Steps: {self.current_steps}", True, WHITE)
        self.screen.blit(steps_text, (10, stats_y + 40))

        # Overall stats
        avg_score = self.total_score / max(1, self.total_episodes)
        avg_text = self.small_font.render(f"Avg: {avg_score:.2f}", True, WHITE)
        self.screen.blit(avg_text, (200, stats_y))

        best_text = self.small_font.render(f"Best: {self.best_score}", True, WHITE)
        self.screen.blit(best_text, (200, stats_y + 40))

        episode_text = self.small_font.render(f"Episode: {self.total_episodes}", True, WHITE)
        self.screen.blit(episode_text, (350, stats_y))

        # Model info
        model_text = self.small_font.render("Stable-Baselines3 PPO", True, GREEN)
        self.screen.blit(model_text, (350, stats_y + 40))

        pygame.display.flip()

    def run_episode(self):
        """Run one episode"""
        state = self.game.reset()
        done = False
        self.current_score = 0
        self.current_steps = 0

        while not done:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    if event.key == pygame.K_SPACE:
                        # Pause
                        paused = True
                        while paused:
                            for e in pygame.event.get():
                                if e.type == pygame.QUIT:
                                    return False
                                if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                                    paused = False

            # Get observation
            obs = self._get_obs(state)

            # Get action from model
            action, _states = self.model.predict(obs, deterministic=True)

            # Step game
            state, reward, done = self.game.step(int(action))

            self.current_score = state.get('score', 0)
            self.current_steps += 1

            # Render
            self.render_game(state)
            self.clock.tick(self.fps)

            # Stop if too long
            if self.current_steps >= 1000:
                done = True

        # Update stats
        self.total_episodes += 1
        self.total_score += self.current_score
        self.best_score = max(self.best_score, self.current_score)

        print(f"Episode {self.total_episodes}: Score = {self.current_score}, Steps = {self.current_steps}")

        return True

    def run(self, num_episodes=10):
        """Run multiple episodes"""
        print("\n" + "=" * 70)
        print("VISUAL DEMO: STABLE-BASELINES3 SNAKE AGENT")
        print("=" * 70)
        print("Controls:")
        print("  SPACE - Pause/Resume")
        print("  ESC   - Quit")
        print()
        print(f"Running {num_episodes} episodes...")
        print("-" * 70)

        for episode in range(num_episodes):
            if not self.run_episode():
                break

        pygame.quit()

        # Final stats
        print()
        print("=" * 70)
        print("DEMO COMPLETE")
        print("=" * 70)
        print(f"Episodes: {self.total_episodes}")
        print(f"Average Score: {self.total_score / max(1, self.total_episodes):.2f}")
        print(f"Best Score: {self.best_score}")
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='snake_ppo_sb3.zip',
                       help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to run')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second')
    parser.add_argument('--cell-size', type=int, default=30,
                       help='Cell size in pixels')
    args = parser.parse_args()

    demo = VisualSnakeDemo(
        model_path=args.model,
        cell_size=args.cell_size,
        fps=args.fps
    )

    demo.run(num_episodes=args.episodes)
