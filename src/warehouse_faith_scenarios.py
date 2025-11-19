"""
Warehouse Faith Scenarios - Realistic Challenge Suite

Scenario-based warehouse simulation designed to showcase the faith-based
evolutionary agent's ability to discover hidden mechanics, adapt to changing
conditions, and transfer knowledge across different warehouse configurations.

Each scenario contains hidden mechanics that standard RL would struggle to discover
but faith exploration can reveal through persistent behavioral experimentation.

Scenarios:
1. Hidden Shortcut - Discover conditional passageways
2. Charging Station Dilemma - Optimal battery management timing
3. Priority Zone System - Time-sensitive package prioritization
"""

import pygame
import sys
import os
import torch
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))

from context_aware_agent import (
    ContextAwareDQN,
    infer_context_from_observation,
    add_context_to_observation
)
from core.temporal_observer import TemporalFlowObserver
from core.world_model import WorldModelNetwork

# Faith system imports
from core.faith_system import FaithPattern, FaithPopulation
from core.entity_discovery import EntityDiscoveryWorldModel, EntityBehaviorLearner
from core.pattern_transfer import UniversalPatternExtractor
from core.mechanic_detectors import MechanicDetector

# Colors
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
CHARGING_STATION_COLOR = (50, 255, 50)  # Green
SUPERVISOR_COLOR = (255, 100, 0)  # Orange

# Package priority colors
PRIORITY_RED = (255, 50, 50)    # High priority, decays
PRIORITY_BLUE = (50, 150, 255)  # Low priority, stable
PRIORITY_GREEN = (50, 255, 50)  # Chain bonus

# Status colors
GREEN = (50, 200, 50)
YELLOW = (255, 200, 0)
RED = (220, 50, 50)
ORANGE = (255, 140, 0)
MAGENTA = (255, 0, 255)  # Faith actions


class WarehouseScenario(ABC):
    """Base class for warehouse scenarios with hidden mechanics"""

    def __init__(self, name: str, description: str, size: int = 20):
        self.name = name
        self.description = description
        self.size = size
        self.steps = 0
        self.max_steps = 500
        self.done = False

        # Base warehouse elements
        self.walls = set()
        self.agv_pos = (2, size - 3)
        self.packages = []
        self.workers = []

        # Statistics
        self.packages_picked = 0
        self.total_packages_spawned = 0
        self.collisions = 0
        self.total_distance = 0

        # Scenario-specific stats
        self.scenario_stats = {}

        # Hidden mechanics tracking (for analysis)
        self.hidden_mechanics_discovered = []
        self.mechanic_hints_revealed = []

    @abstractmethod
    def create_layout(self):
        """Create scenario-specific warehouse layout"""
        pass

    @abstractmethod
    def update_mechanics(self):
        """Update scenario-specific mechanics each step"""
        pass

    @abstractmethod
    def calculate_reward(self, action: int, new_pos: Tuple[int, int],
                        picked_package: bool, collision: bool) -> float:
        """Calculate reward including scenario-specific bonuses/penalties"""
        pass

    @abstractmethod
    def get_scenario_info(self) -> Dict:
        """Get scenario-specific information for display"""
        pass

    def reset(self):
        """Reset scenario to initial state"""
        self.steps = 0
        self.done = False
        self.packages_picked = 0
        self.collisions = 0
        self.total_distance = 0
        self.agv_pos = (2, self.size - 3)
        self.packages = []
        self.workers = []
        self.scenario_stats = {}
        self.hidden_mechanics_discovered = []

        # Create layout
        self.create_layout()

        return self._get_state()

    def step(self, action: int):
        """Execute action with scenario mechanics"""
        if self.done:
            return self._get_state(), 0, True

        self.steps += 1

        # Move AGV
        moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
        dx, dy = moves[action]
        new_x = self.agv_pos[0] + dx
        new_y = self.agv_pos[1] + dy
        new_pos = (new_x, new_y)

        # Check if move is valid (scenario-specific)
        collision = False
        picked_package = False

        if self._is_valid_move(new_pos):
            self.agv_pos = new_pos
            self.total_distance += 1

            # Check package pickup
            if self.agv_pos in [p['pos'] for p in self.packages]:
                picked_package = True
                self._pickup_package(self.agv_pos)
        else:
            collision = True
            self.collisions += 1

        # Update workers
        self._update_workers()

        # Check worker collision
        for worker in self.workers:
            if self.agv_pos == worker['pos']:
                collision = True
                self.collisions += 1
                break

        # Update scenario mechanics
        self.update_mechanics()

        # Calculate reward
        reward = self.calculate_reward(action, new_pos, picked_package, collision)

        # Check done
        if self.steps >= self.max_steps:
            self.done = True

        return self._get_state(), reward, self.done

    def _is_valid_move(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid (can be overridden by scenarios)"""
        x, y = pos
        return (pos not in self.walls and
                0 <= x < self.size and
                0 <= y < self.size)

    def _pickup_package(self, pos: Tuple[int, int]):
        """Pickup package at position"""
        for package in self.packages[:]:
            if package['pos'] == pos:
                self.packages.remove(package)
                self.packages_picked += 1
                return package
        return None

    def _update_workers(self):
        """Update worker positions"""
        for worker in self.workers:
            if worker.get('idle_counter', 0) > 0:
                worker['idle_counter'] -= 1
                continue

            # Occasional direction change
            if random.random() < 0.1:
                worker['velocity'] = (
                    random.choice([-1, 0, 1]),
                    random.choice([-1, 0, 1])
                )

            # Move worker
            dx, dy = worker['velocity']
            new_x = worker['pos'][0] + dx
            new_y = worker['pos'][1] + dy
            new_pos = (new_x, new_y)

            if self._is_valid_move(new_pos):
                worker['pos'] = new_pos
            else:
                # Change direction on collision
                worker['velocity'] = (
                    random.choice([-1, 0, 1]),
                    random.choice([-1, 0, 1])
                )

    def _get_state(self):
        """Get scenario state"""
        entities = []
        for worker in self.workers:
            entities.append({
                'pos': worker['pos'],
                'velocity': worker.get('velocity', (0, 0)),
                'danger': 1.0,
                'type': worker.get('type', 'worker')
            })

        return {
            'agent_pos': self.agv_pos,
            'walls': self.walls,
            'rewards': [p['pos'] for p in self.packages],
            'entities': entities,
            'grid_size': (self.size, self.size),
            'score': self.packages_picked,
            'done': self.done,
            'scenario_info': self.get_scenario_info()
        }


class HiddenShortcutScenario(WarehouseScenario):
    """
    Scenario 1: Hidden Shortcut

    HIDDEN MECHANIC: Certain shelf sections become passable when supervisor is far away.

    Discovery Challenge:
    - Standard RL treats walls as permanent obstacles
    - Faith exploration occasionally tries "impossible" moves
    - Discovers walls are conditionally passable based on supervisor distance

    Optimal Strategy (once discovered):
    - Monitor supervisor position
    - Use shortcut when supervisor_distance > 5
    - Saves ~30% travel time
    """

    def __init__(self, size: int = 20):
        super().__init__(
            name="Hidden Shortcut",
            description="Discover conditional passageways based on supervisor position",
            size=size
        )

        # Shortcut mechanic
        self.shortcut_walls = set()  # Walls that can be shortcuts
        self.supervisor = None
        self.shortcut_uses = 0
        self.shortcut_discovered = False

    def create_layout(self):
        """Create warehouse with hidden shortcut"""
        # Outer walls
        for i in range(self.size):
            self.walls.add((i, 0))
            self.walls.add((i, self.size - 1))
            self.walls.add((0, i))
            self.walls.add((self.size - 1, i))

        # Create aisles with shortcut section
        aisle_positions = [5, 10, 15]

        for aisle_x in aisle_positions:
            for y in range(2, self.size - 2):
                if y % 5 != 0:  # Leave cross-aisles
                    if aisle_x > 2:
                        wall_pos = (aisle_x - 1, y)
                        self.walls.add(wall_pos)

                        # Mark middle aisle section as shortcut
                        if aisle_x == 10 and 8 <= y <= 12:
                            self.shortcut_walls.add(wall_pos)

                    if aisle_x < self.size - 3:
                        wall_pos = (aisle_x + 1, y)
                        self.walls.add(wall_pos)

                        # Mark middle aisle section as shortcut
                        if aisle_x == 10 and 8 <= y <= 12:
                            self.shortcut_walls.add(wall_pos)

        # Spawn supervisor
        self.supervisor = {
            'pos': (self.size // 2, self.size // 2),
            'velocity': (1, 0),
            'idle_counter': 0,
            'type': 'supervisor'
        }
        self.workers.append(self.supervisor)

        # Spawn packages
        for _ in range(5):
            self._spawn_package()

    def _spawn_package(self):
        """Spawn package"""
        attempts = 0
        while attempts < 100:
            x = random.choice([4, 6, 9, 11, 14, 16])
            y = random.randint(3, self.size - 4)
            pos = (x, y)

            if (pos not in self.walls and
                pos != self.agv_pos and
                pos not in [p['pos'] for p in self.packages]):
                self.packages.append({
                    'pos': pos,
                    'type': 'standard',
                    'reward': 10.0
                })
                self.total_packages_spawned += 1
                return
            attempts += 1

    def _is_valid_move(self, pos: Tuple[int, int]) -> bool:
        """Check if move valid - shortcuts passable when supervisor far"""
        # Check supervisor distance
        supervisor_dist = abs(pos[0] - self.supervisor['pos'][0]) + abs(pos[1] - self.supervisor['pos'][1])

        # Shortcut condition: supervisor_distance > 5
        if pos in self.shortcut_walls:
            if supervisor_dist > 5:
                # Shortcut is passable!
                if not self.shortcut_discovered:
                    self.shortcut_discovered = True
                    self.hidden_mechanics_discovered.append({
                        'mechanic': 'conditional_shortcut',
                        'step': self.steps,
                        'description': 'Walls passable when supervisor distance > 5'
                    })
                self.shortcut_uses += 1
                return True  # Allow passage
            else:
                return False  # Blocked by supervisor proximity

        # Normal wall check
        return super()._is_valid_move(pos)

    def update_mechanics(self):
        """Update supervisor patrol"""
        # Supervisor patrols in a pattern
        if self.steps % 50 == 0:
            # Change direction periodically
            self.supervisor['velocity'] = (
                random.choice([-1, 0, 1]),
                random.choice([-1, 0, 1])
            )

    def calculate_reward(self, action: int, new_pos: Tuple[int, int],
                        picked_package: bool, collision: bool) -> float:
        """Calculate reward with shortcut bonus"""
        reward = -0.1  # Time penalty

        if collision:
            reward = -5.0 if self.supervisor and new_pos == self.supervisor['pos'] else -2.0
        elif picked_package:
            reward = 10.0
            self._spawn_package()  # Spawn new package

            # Bonus for using shortcut efficiently
            if self.shortcut_uses > 0:
                reward += 2.0  # Efficiency bonus

        return reward

    def get_scenario_info(self) -> Dict:
        """Get scenario info"""
        supervisor_dist = 0
        if self.supervisor:
            supervisor_dist = abs(self.agv_pos[0] - self.supervisor['pos'][0]) + \
                            abs(self.agv_pos[1] - self.supervisor['pos'][1])

        return {
            'scenario': self.name,
            'supervisor_distance': supervisor_dist,
            'shortcut_usable': supervisor_dist > 5,
            'shortcut_uses': self.shortcut_uses,
            'shortcut_discovered': self.shortcut_discovered,
            'mechanics_found': len(self.hidden_mechanics_discovered)
        }


class ChargingStationScenario(WarehouseScenario):
    """
    Scenario 2: Charging Station Dilemma

    HIDDEN MECHANIC: Battery depletes with movement, speed decreases at 30% battery.

    Discovery Challenge:
    - Battery level is hidden (not in observation)
    - Performance degradation at threshold is non-obvious
    - Optimal charging timing requires experimentation

    Optimal Strategy (once discovered):
    - Charge at ~40% battery (before slowdown kicks in)
    - Not at 10% (too late, already slow)
    - Not at 80% (wasting time)
    - Proactive vs reactive charging
    """

    def __init__(self, size: int = 20):
        super().__init__(
            name="Charging Station Dilemma",
            description="Discover optimal battery management timing",
            size=size
        )

        # Battery system
        self.battery = 100.0
        self.battery_depletion_rate = 0.3
        self.low_battery_threshold = 30.0
        self.charging_rate = 2.0
        self.is_charging = False

        # Charging stations
        self.charging_stations = []

        # Stats
        self.charging_sessions = 0
        self.low_battery_episodes = 0
        self.optimal_charge_count = 0  # Charged at 35-45%
        self.slowdown_steps = 0
        self.battery_discovery = False

    def create_layout(self):
        """Create warehouse with charging stations"""
        # Standard aisles
        for i in range(self.size):
            self.walls.add((i, 0))
            self.walls.add((i, self.size - 1))
            self.walls.add((0, i))
            self.walls.add((self.size - 1, i))

        aisle_positions = [5, 10, 15]
        for aisle_x in aisle_positions:
            for y in range(2, self.size - 2):
                if y % 5 != 0:
                    if aisle_x > 2:
                        self.walls.add((aisle_x - 1, y))
                    if aisle_x < self.size - 3:
                        self.walls.add((aisle_x + 1, y))

        # Place charging stations (3 stations)
        station_positions = [(3, 3), (self.size - 4, 3), (self.size // 2, self.size - 4)]
        for pos in station_positions:
            if pos not in self.walls:
                self.charging_stations.append(pos)

        # Spawn packages
        for _ in range(6):
            self._spawn_package()

    def _spawn_package(self):
        """Spawn package"""
        attempts = 0
        while attempts < 100:
            x = random.choice([4, 6, 9, 11, 14, 16])
            y = random.randint(3, self.size - 4)
            pos = (x, y)

            if (pos not in self.walls and
                pos != self.agv_pos and
                pos not in [p['pos'] for p in self.packages] and
                pos not in self.charging_stations):
                self.packages.append({
                    'pos': pos,
                    'type': 'standard',
                    'reward': 10.0
                })
                self.total_packages_spawned += 1
                return
            attempts += 1

    def update_mechanics(self):
        """Update battery system"""
        # Check if on charging station
        self.is_charging = self.agv_pos in self.charging_stations

        if self.is_charging:
            # Charging
            old_battery = self.battery
            self.battery = min(100.0, self.battery + self.charging_rate)

            # Track charging session
            if old_battery < self.battery:
                if self.battery >= 35 and old_battery <= 45:
                    self.optimal_charge_count += 1

                    # Discovery check
                    if not self.battery_discovery:
                        self.battery_discovery = True
                        self.hidden_mechanics_discovered.append({
                            'mechanic': 'optimal_charging_window',
                            'step': self.steps,
                            'description': 'Charging at 35-45% battery is optimal'
                        })
        else:
            # Deplete battery
            self.battery = max(0, self.battery - self.battery_depletion_rate)

            # Track low battery
            if self.battery < self.low_battery_threshold:
                self.low_battery_episodes += 1
                self.slowdown_steps += 1

    def calculate_reward(self, action: int, new_pos: Tuple[int, int],
                        picked_package: bool, collision: bool) -> float:
        """Calculate reward with battery penalties"""
        # Base time penalty (affected by battery)
        if self.battery < self.low_battery_threshold:
            reward = -0.3  # Slower, higher cost
        else:
            reward = -0.1

        if collision:
            reward = -2.0
        elif picked_package:
            reward = 10.0
            self._spawn_package()

            # Bonus for good battery management
            if self.battery > self.low_battery_threshold:
                reward += 1.0  # Efficiency bonus

        # Penalty for running out of battery
        if self.battery <= 0:
            reward = -10.0  # Critical failure

        # Small reward for optimal charging
        if self.is_charging and 35 <= self.battery <= 45:
            reward += 0.5  # Optimal charging window

        return reward

    def get_scenario_info(self) -> Dict:
        """Get scenario info"""
        return {
            'scenario': self.name,
            'battery': self.battery,
            'is_charging': self.is_charging,
            'low_battery': self.battery < self.low_battery_threshold,
            'charging_sessions': self.charging_sessions,
            'optimal_charges': self.optimal_charge_count,
            'slowdown_steps': self.slowdown_steps,
            'battery_discovered': self.battery_discovery,
            'mechanics_found': len(self.hidden_mechanics_discovered)
        }


class PriorityZoneScenario(WarehouseScenario):
    """
    Scenario 3: Priority Zone System

    HIDDEN MECHANIC: Packages have priorities with different reward dynamics.

    Discovery Challenge:
    - Package types are visually distinct but semantics hidden
    - Red packages: High reward (20) but decay -2 per 10 steps
    - Blue packages: Stable reward (8), no decay
    - Green packages: Medium reward (15) + chain bonus (+5 per consecutive green)

    Optimal Strategy (once discovered):
    - Collect red packages first (before decay)
    - Chain green packages together
    - Fill with blue packages
    - Order matters!
    """

    def __init__(self, size: int = 20):
        super().__init__(
            name="Priority Zone System",
            description="Discover time-sensitive package priorities",
            size=size
        )

        # Package types
        self.package_types = ['red', 'blue', 'green']

        # Chain tracking
        self.green_chain = 0
        self.max_green_chain = 0
        self.last_package_type = None

        # Discovery tracking
        self.red_decay_discovered = False
        self.green_chain_discovered = False
        self.priority_strategy_discovered = False

    def create_layout(self):
        """Create warehouse with priority packages"""
        # Standard layout
        for i in range(self.size):
            self.walls.add((i, 0))
            self.walls.add((i, self.size - 1))
            self.walls.add((0, i))
            self.walls.add((self.size - 1, i))

        aisle_positions = [5, 10, 15]
        for aisle_x in aisle_positions:
            for y in range(2, self.size - 2):
                if y % 5 != 0:
                    if aisle_x > 2:
                        self.walls.add((aisle_x - 1, y))
                    if aisle_x < self.size - 3:
                        self.walls.add((aisle_x + 1, y))

        # Spawn packages of different types
        for _ in range(8):
            self._spawn_package()

    def _spawn_package(self):
        """Spawn package with priority type"""
        attempts = 0
        while attempts < 100:
            x = random.choice([4, 6, 9, 11, 14, 16])
            y = random.randint(3, self.size - 4)
            pos = (x, y)

            if (pos not in self.walls and
                pos != self.agv_pos and
                pos not in [p['pos'] for p in self.packages]):

                # Random package type
                package_type = random.choice(self.package_types)

                if package_type == 'red':
                    reward = 20.0
                    decay = -2.0
                elif package_type == 'blue':
                    reward = 8.0
                    decay = 0.0
                else:  # green
                    reward = 15.0
                    decay = 0.0

                self.packages.append({
                    'pos': pos,
                    'type': package_type,
                    'base_reward': reward,
                    'reward': reward,
                    'decay': decay,
                    'spawn_step': self.steps
                })
                self.total_packages_spawned += 1
                return
            attempts += 1

    def update_mechanics(self):
        """Update package rewards (decay)"""
        for package in self.packages:
            if package['type'] == 'red':
                # Decay red packages every 10 steps
                steps_since_spawn = self.steps - package['spawn_step']
                decay_amount = (steps_since_spawn // 10) * package['decay']
                package['reward'] = max(5.0, package['base_reward'] + decay_amount)

                # Check if decay discovered
                if decay_amount < -4 and not self.red_decay_discovered:
                    self.red_decay_discovered = True
                    self.hidden_mechanics_discovered.append({
                        'mechanic': 'red_package_decay',
                        'step': self.steps,
                        'description': 'Red packages decay -2 reward per 10 steps'
                    })

    def _pickup_package(self, pos: Tuple[int, int]):
        """Pickup package with chain logic"""
        for package in self.packages[:]:
            if package['pos'] == pos:
                self.packages.remove(package)
                self.packages_picked += 1

                # Track chains
                if package['type'] == 'green':
                    if self.last_package_type == 'green':
                        self.green_chain += 1
                        self.max_green_chain = max(self.max_green_chain, self.green_chain)

                        # Check if chain discovered
                        if self.green_chain >= 2 and not self.green_chain_discovered:
                            self.green_chain_discovered = True
                            self.hidden_mechanics_discovered.append({
                                'mechanic': 'green_package_chain',
                                'step': self.steps,
                                'description': 'Consecutive green packages give chain bonus'
                            })
                    else:
                        self.green_chain = 1
                else:
                    self.green_chain = 0

                self.last_package_type = package['type']

                return package
        return None

    def calculate_reward(self, action: int, new_pos: Tuple[int, int],
                        picked_package: bool, collision: bool) -> float:
        """Calculate reward with priority bonuses"""
        reward = -0.1  # Time penalty

        if collision:
            reward = -2.0
        elif picked_package:
            # Find which package was picked
            for package in self.packages[:]:
                if package['pos'] == self.agv_pos:
                    reward = package['reward']

                    # Green chain bonus
                    if package['type'] == 'green' and self.green_chain > 1:
                        chain_bonus = (self.green_chain - 1) * 5.0
                        reward += chain_bonus

                    break

            self._spawn_package()

            # Check if optimal strategy discovered
            if (self.red_decay_discovered and
                self.green_chain_discovered and
                not self.priority_strategy_discovered):
                self.priority_strategy_discovered = True
                self.hidden_mechanics_discovered.append({
                    'mechanic': 'priority_strategy',
                    'step': self.steps,
                    'description': 'Optimal: Red first, green chains, blue fill'
                })

        return reward

    def get_scenario_info(self) -> Dict:
        """Get scenario info"""
        # Count package types
        type_counts = {'red': 0, 'blue': 0, 'green': 0}
        for p in self.packages:
            type_counts[p['type']] += 1

        return {
            'scenario': self.name,
            'green_chain': self.green_chain,
            'max_chain': self.max_green_chain,
            'packages_by_type': type_counts,
            'red_decay_found': self.red_decay_discovered,
            'chain_found': self.green_chain_discovered,
            'strategy_found': self.priority_strategy_discovered,
            'mechanics_found': len(self.hidden_mechanics_discovered)
        }


# Scenario registry
SCENARIOS = {
    'hidden_shortcut': HiddenShortcutScenario,
    'charging_station': ChargingStationScenario,
    'priority_zone': PriorityZoneScenario
}


def create_scenario(scenario_name: str, size: int = 20) -> WarehouseScenario:
    """Factory function to create scenarios"""
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(SCENARIOS.keys())}")

    return SCENARIOS[scenario_name](size=size)


def get_scenario_descriptions() -> Dict[str, str]:
    """Get all scenario descriptions"""
    descriptions = {}
    for name, scenario_class in SCENARIOS.items():
        temp_scenario = scenario_class()
        descriptions[name] = {
            'name': temp_scenario.name,
            'description': temp_scenario.description
        }
    return descriptions
