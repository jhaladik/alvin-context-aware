# Differences: faith_visual_games.py vs expanded_faith_visual_games.py

## Summary: MINIMAL CHANGES - Only Observer & Dimensions

The expanded visual demo is **identical** to the original except for:
1. Observer type (92-dim → 180-dim)
2. Network dimensions (95 → 183)
3. Planning horizon default (5 → 20)
4. Visual enhancements (8 rays → 16 rays display)

**All game logic and rendering is EXACTLY THE SAME!** ✅

---

## Key Changes

### 1. Import Statement (line 24)
```python
# Before:
from core.temporal_observer import TemporalFlowObserver

# After:
from core.expanded_temporal_observer import ExpandedTemporalObserver  # EXPANDED: 180 dims!
```

### 2. Class Name (line 56)
```python
# Before:
class FaithVisualRunner:

# After:
class ExpandedFaithVisualRunner:
```

### 3. Planning Horizon Default (line 62)
```python
# Before:
def __init__(self, model_path, cell_size=25, use_planning=True, planning_freq=0.3,
             planning_horizon=5, faith_freq=0.1):

# After:
def __init__(self, model_path, cell_size=25, use_planning=True, planning_freq=0.3,
             planning_horizon=20, faith_freq=0.1):  # EXPANDED: 20 steps vs 5
```

### 4. Observer Initialization (line 73)
```python
# Before:
self.observer = TemporalFlowObserver()

# After:
# EXPANDED: 16 rays × 15 tiles, multi-scale temporal (180 dims)
self.observer = ExpandedTemporalObserver(num_rays=16, ray_length=15, verbose=False)
```

### 5. Network Dimensions (lines 90-94)
```python
# Before:
self.agent = ContextAwareDQN(obs_dim=95, action_dim=4)
self.world_model = WorldModelNetwork(state_dim=95, action_dim=4)
self.entity_world_model = EntityDiscoveryWorldModel(obs_dim=95, action_dim=4, max_entity_types=20)

# After:
# EXPANDED: 183 dims (180 observer + 3 context)
self.agent = ContextAwareDQN(obs_dim=183, action_dim=4)
self.world_model = WorldModelNetwork(state_dim=183, action_dim=4)
self.entity_world_model = EntityDiscoveryWorldModel(obs_dim=183, action_dim=4, max_entity_types=20)
```

### 6. Info Display Updates (lines 129-138)
```python
# Before:
print(f"  Input: 95-dim (92 temporal + 3 context)")

# After:
print(f"  Input: 183-dim (180 EXPANDED temporal + 3 context)")
print(f"  Observer: 16 rays × 15 tiles (vs 8 rays × 10 tiles baseline)")
print(f"  Multi-scale temporal: Micro (5) + Meso (20) + Macro (50) frames")
print(f"  Planning horizon: {planning_horizon} steps (vs 5 baseline)")
```

### 7. Window Width (line 219)
```python
# Before:
width = self.grid_size * self.cell_size + 300

# After:
width = self.grid_size * self.cell_size + 320  # Extra space for EXPANDED info
```

### 8. Window Title (line 222)
```python
# Before:
pygame.display.set_caption(f'Faith-Based Agent - {self.game_name}')

# After:
pygame.display.set_caption(f'EXPANDED Faith-Based Agent - {self.game_name}')
```

### 9. Temporal Info Extraction (lines 224-236)
```python
# Before:
def _update_temporal_info(self):
    if self.current_obs is not None and len(self.current_obs) >= 92:
        self.reward_direction = (self.current_obs[46], self.current_obs[47])
        if len(self.current_obs) > 90:
            self.danger_trend = self.current_obs[88]
            self.progress_rate = self.current_obs[90]

# After:
def _update_temporal_info(self):
    # EXPANDED: 180-dim observation
    if self.current_obs is not None and len(self.current_obs) >= 180:
        # Reward direction (adjusted indices for 16 rays)
        # With 16 rays: ray features = 16*3 = 48, next is context features
        self.reward_direction = (self.current_obs[48], self.current_obs[49])

        # Danger trend and progress rate (in macro patterns section)
        if len(self.current_obs) > 170:
            self.danger_trend = self.current_obs[170]
            self.progress_rate = self.current_obs[172]
```

### 10. Ray Visualization (lines 467-527)
```python
# Before:
def _draw_detection_rays(self, center_x, center_y):
    """Draw rays showing entity and wall detection in 8 directions"""
    if self.current_obs is None or len(self.current_obs) < 48:
        return

    # Ray directions: N, NE, E, SE, S, SW, W, NW
    ray_dirs = [
        (0, -1), (1, -1), (1, 0), (1, 1),
        (0, 1), (-1, 1), (-1, 0), (-1, -1)
    ]

    max_ray_length = self.cell_size * 4

# After:
def _draw_detection_rays(self, center_x, center_y):
    """Draw rays showing entity and wall detection in 16 directions (EXPANDED)"""
    # EXPANDED: 180-dim observation with 16 rays
    if self.current_obs is None or len(self.current_obs) < 48:
        return

    # EXPANDED: 16 ray directions (22.5° apart)
    ray_dirs = []
    for i in range(16):
        angle = i * (2 * np.pi / 16)
        ray_dirs.append((np.cos(angle), np.sin(angle)))

    max_ray_length = self.cell_size * 5  # EXPANDED: longer range (15 tiles)
```

### 11. Info Panel (lines 541-547)
```python
# Added EXPANDED indicator below game name:
expanded_label = self.tiny_font.render('EXPANDED (180 dims, 20-step)', True, TEAL)
self.screen.blit(expanded_label, (panel_x, y))
```

### 12. Startup Message (lines 825-843)
```python
# Added expanded capabilities section:
print("EXPANDED CAPABILITIES:")
print("  Spatial-Temporal Observer: 16 rays × 15 tiles (180 dims vs 92)")
print("  Multi-Scale Temporal: Micro (5) + Meso (20) + Macro (50) frames")
print("  Ghost Mode Detection: Chase/Scatter/Random behavior patterns")
print("  Extended Planning: 20-step lookahead (vs 5 baseline)")
print()
```

### 13. Visual Indicators (line 853)
```python
# Updated:
print("  Gray rays = Wall detection (16 rays in EXPANDED!)")
```

### 14. Argparse Default (line 862)
```python
# Before:
parser.add_argument('--planning-horizon', type=int, default=5, help='Planning horizon (steps)')

# After:
parser.add_argument('--planning-horizon', type=int, default=20, help='Planning horizon (steps) - EXPANDED: 20 vs 5')
```

---

## What's EXACTLY THE SAME ✅

**All game rendering and logic is identical:**
- ✅ `draw_snake()` method - NO CHANGES
- ✅ `draw_pacman()` method - NO CHANGES
- ✅ `draw_dungeon()` method - NO CHANGES
- ✅ `draw_info_panel()` structure - NO CHANGES (just extended info)
- ✅ `run()` game loop - NO CHANGES
- ✅ Faith action selection - NO CHANGES
- ✅ Planning logic - NO CHANGES
- ✅ Pygame rendering - NO CHANGES
- ✅ Keyboard controls - NO CHANGES
- ✅ Auto-reset on done - NO CHANGES

---

## Visual Enhancements

The EXPANDED version shows:
1. **16 detection rays** (vs 8) - more detailed spatial awareness visualization
2. **Longer ray range** (5 cells vs 4) - shows extended vision
3. **EXPANDED label** on screen - indicates 180-dim, 20-step mode
4. **Same color coding**:
   - MAGENTA = Faith action
   - CYAN = Planning action
   - WHITE/YELLOW/GREEN = Reactive action

---

## Usage

### Run with EXPANDED agent checkpoint:
```bash
cd src
python expanded_faith_visual_games.py \
    --model checkpoints/faith_evolution_20251119_211514_best_policy.pth \
    --speed 10 \
    --planning-freq 0.3 \
    --planning-horizon 20 \
    --faith-freq 0.1
```

### Compare to baseline:
```bash
# Baseline (92-dim, 5-step)
python faith_visual_games.py \
    --model checkpoints/<baseline_checkpoint>_policy.pth \
    --speed 10

# EXPANDED (180-dim, 20-step)
python expanded_faith_visual_games.py \
    --model checkpoints/faith_evolution_20251119_211514_best_policy.pth \
    --speed 10
```

---

## Conclusion

We created a **clean, minimal extension** of the visual demo:
- Same game logic, rendering, controls
- Only expanded observation and planning
- Visual enhancements show 16 rays instead of 8
- Clean indicator showing EXPANDED mode

This allows **direct visual comparison** between baseline and expanded agents! 🎮✨
