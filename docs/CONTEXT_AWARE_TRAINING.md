# Context-Aware Training System

## Problem: Spurious Correlations in Transfer Learning

### The Issue
When training on random environments with entities and rewards placed randomly, the agent learns **spurious correlations**:

- **Training**: Entities often spawn near rewards (random placement)
- **Agent learns**: "Reward proximity = Entity proximity = DANGER"
- **Deployment (Snake)**: No entities, but agent still avoids rewards

**Result**: Snake performance degrades (0.40 → 0.00) while Pac-Man/Dungeon improve.

### Root Cause
This is a **distribution shift** problem:
- Training distribution: Mixed entity counts (0-6)
- Test distribution: Fixed entity counts (0 for Snake, 4 for Pac-Man)
- Agent cannot distinguish between contexts

## Solution: Context-Aware Architecture

### Key Insight
> "You already know what environment you are. Add strategy level to training and mix various environments."
> - User's elegant solution

Instead of treating all environments as identical, we:
1. **Augment observations** with 3-dimensional context vector
2. **Train on mixed scenarios** (0 entities, 2-3 entities, 4+ entities)
3. **Infer context at test time** from observation features

### Context Vector (3-dimensional)
```python
[1, 0, 0]  # SNAKE mode: No entities (pure collection)
[0, 1, 0]  # BALANCED mode: 2-3 entities (tactical gameplay)
[0, 0, 1]  # SURVIVAL mode: 4+ entities (high threat)
```

### Architecture
```
Input: 95 features = 92 temporal + 3 context
       ↓
Perception Network (128 hidden units)
       ↓
4 Hierarchical Q-heads:
- Survival Q-head
- Avoidance Q-head
- Positioning Q-head
- Collection Q-head
       ↓
Combined Q = weighted sum
```

## Files

### 1. `context_aware_agent.py`
Core model architecture with context support.

**Key Classes**:
- `ContextAwareDQN`: 95-dim input hierarchical DQN
- `infer_context_from_observation()`: Detect environment type from obs
- `add_context_to_observation()`: Concatenate context vector

**Context Inference Logic**:
```python
def infer_context_from_observation(obs):
    # Count detected entities from 8 rays
    entity_count = 0
    for i in range(8):
        entity_dist = obs[i * 3]
        if entity_dist < 0.9:  # Entity detected
            entity_count += 1

    # Map to context vector
    if entity_count == 0:
        return [1.0, 0.0, 0.0]  # Snake
    elif entity_count <= 3:
        return [0.0, 1.0, 0.0]  # Balanced
    else:
        return [0.0, 0.0, 1.0]  # Survival
```

### 2. `train_context_aware.py`
Training script with mixed scenarios.

**Context Distribution**:
- 30% Snake mode (0 entities)
- 50% Balanced mode (2-3 entities)
- 20% Survival mode (4+ entities)

**Training Process**:
1. Sample context based on distribution
2. Create environment with appropriate num_entities
3. Concatenate context vector to all observations
4. Train policy network + world model
5. Track per-context performance

**Usage**:
```bash
# Train for 5000 episodes
python train_context_aware.py --episodes 5000

# Custom settings
python train_context_aware.py --episodes 2000 --log-every 50
```

**Output**:
```
Episode 100/5000
  Avg Reward (100): 12.34
  Avg Length (100): 234.5
  Policy Loss: 0.1234
  World Model Loss: 0.5678
  Epsilon: 0.900
  Buffer Size: 10000
  Steps: 23450
  Context Distribution:
    snake   :   28 episodes (28.0%) - avg reward:  15.23
    balanced:   52 episodes (52.0%) - avg reward:  11.45
    survival:   20 episodes (20.0%) - avg reward:   8.67
```

### 3. `test_context_aware.py`
Command-line testing on Snake, Pac-Man, and Dungeon.

**Features**:
- Tests agent on all three games
- Tracks context detection accuracy
- Reports per-game performance
- Validates context adaptation

**Usage**:
```bash
# Test all games (50 episodes each)
python test_context_aware.py checkpoints/context_aware_best_policy.pth

# Test specific game
python test_context_aware.py model.pth --game snake --episodes 100
```

**Expected Output**:
```
SNAKE TEST RESULTS
Episodes: 50
Average Score: 8.50 ± 2.34
Max Score: 15
Min Score: 3
Average Steps: 234.5

Context Distribution:
  snake   :  11234 steps ( 95.6%)  ✓ GOOD
  balanced:    456 steps (  3.9%)
  survival:     78 steps (  0.5%)

CONTEXT ADAPTATION CHECK:
  Snake game: 95.6% 'snake' context detected
    ✓ GOOD: Agent correctly detects no-entity context
```

### 4. `context_aware_visual_games.py`
Visual testing with pygame.

**Features**:
- Real-time context detection display
- Color-coded context modes (GREEN=Snake, ORANGE=Balanced, RED=Survival)
- Entity detection rays
- Reward direction arrow
- Q-value visualization

**Usage**:
```bash
# Visual test all games
python context_aware_visual_games.py checkpoints/context_aware_best_policy.pth

# Test specific game
python context_aware_visual_games.py model.pth --game snake
```

## Training Workflow

### 1. Train Context-Aware Agent
```bash
cd ml-training/foundation_2d
python train_context_aware.py --episodes 5000 --log-every 100
```

This creates:
- `checkpoints/context_aware_YYYYMMDD_HHMMSS_best_policy.pth`
- `checkpoints/context_aware_YYYYMMDD_HHMMSS_best_world_model.pth`
- `checkpoints/context_aware_YYYYMMDD_HHMMSS_final_policy.pth`
- `checkpoints/context_aware_YYYYMMDD_HHMMSS_final_world_model.pth`

### 2. Test Performance
```bash
# Command-line test
python test_context_aware.py checkpoints/context_aware_YYYYMMDD_HHMMSS_best_policy.pth

# Visual test
python context_aware_visual_games.py checkpoints/context_aware_YYYYMMDD_HHMMSS_best_policy.pth
```

### 3. Validate Context Adaptation
Check that:
- **Snake**: >80% 'snake' context detected
- **Pac-Man**: >60% 'balanced' or 'survival' context detected
- **Dungeon**: >50% 'survival' context detected

## Expected Results

### Before (Temporal Agent)
| Game    | Avg Score | Issue |
|---------|-----------|-------|
| Snake   | 0.00      | Avoids food (spurious correlation) |
| Pac-Man | 3.45      | OK |
| Dungeon | 2.12      | OK |

### After (Context-Aware Agent)
| Game    | Avg Score | Context | Behavior |
|---------|-----------|---------|----------|
| Snake   | 8.50+     | SNAKE (95%+) | Aggressive collection |
| Pac-Man | 4.00+     | BALANCED/SURVIVAL | Tactical avoidance |
| Dungeon | 2.50+     | SURVIVAL | Cautious navigation |

## Key Advantages

1. **No Spurious Correlations**: Agent learns context-dependent behaviors
2. **Adaptive Strategy**: Automatically detects and adapts to environment
3. **Transfer Learning**: Works across different entity densities
4. **Interpretable**: Context vector shows agent's understanding
5. **Foundation Model**: Single model handles multiple game types

## Technical Details

### Input Features (95-dim)
```
Temporal features (92-dim):
  [0:24]   - 8 rays × 3 (entity_dist, entity_danger, wall_dist)
  [24:46]  - Previous ray features (temporal flow)
  [46:48]  - Reward direction (dx, dy)
  [48:50]  - Previous reward direction
  [50:52]  - Agent velocity (dx, dy)
  [52:84]  - Reward history (last 32 rewards)
  [84:92]  - Danger trends, progress rates

Context features (3-dim):
  [92:95]  - Context vector [snake, balanced, survival]
```

### Network Architecture
```
Input (95) → Linear(95, 128) → ReLU
           → Linear(128, 128) → ReLU
           → [Survival Head (64 → 4)]
           → [Avoidance Head (64 → 4)]
           → [Positioning Head (64 → 4)]
           → [Collection Head (64 → 4)]
           → Weighted combination
```

### Priority Weights
```python
survive: 8.0    # High priority
avoid: 4.0      # Medium priority
position: 2.0   # Low priority
collect: 8.0    # High priority (EQUAL to survive!)
```

## Troubleshooting

### Issue: Agent still avoids rewards in Snake
**Check**:
1. Is context correctly detected? (Should be >80% 'snake' mode)
2. Are collection weights high enough? (Should be 8.0)
3. Is environment proximity penalty low? (Should be ≤1.0)

**Fix**:
```bash
# Visual inspection
python context_aware_visual_games.py model.pth --game snake
# Watch context display (should show GREEN "SNAKE" mode)
```

### Issue: Poor performance in all games
**Check**:
1. Training episodes (need 3000+ for convergence)
2. Context distribution (should match target distribution)
3. Replay buffer size (need 50k+ experiences)

**Fix**:
```bash
# Train longer
python train_context_aware.py --episodes 10000
```

### Issue: Context inference incorrect
**Check**:
```python
# In test_context_aware.py output
Context Distribution:
  snake   :  11234 steps ( 95.6%)  # Should be >80% for Snake
  balanced:    456 steps (  3.9%)
  survival:     78 steps (  0.5%)
```

**Fix**: Tune `infer_context_from_observation()` thresholds

## Future Improvements

1. **Curriculum Learning**: Start with Snake, gradually add entities
2. **Meta-Learning**: Learn to adapt quickly to new contexts
3. **Hierarchical Context**: Add sub-contexts (aggressive vs cautious)
4. **Attention Mechanism**: Let agent learn which features matter per context
5. **Multi-Task Learning**: Shared perception, context-specific policies

## Comparison to Other Approaches

### Behavioral Cloning
❌ Copies human moves only
❌ Limited by human skill
❌ No exploration

### Standard RL
❌ Single policy for all contexts
❌ Spurious correlations
❌ Poor transfer learning

### Context-Aware RL (Ours)
✅ Learns context-dependent behaviors
✅ Automatic context detection
✅ Strong transfer learning
✅ Single model, multiple strategies

## Summary

The context-aware architecture solves the spurious correlation problem by:
1. **Training on mixed scenarios** (0, 2-3, 4+ entities)
2. **Explicit context signals** during training
3. **Automatic context inference** at test time
4. **Context-dependent behaviors** learned implicitly

This enables the agent to:
- Play Snake aggressively (collection focus)
- Play Pac-Man tactically (balanced)
- Play Dungeon cautiously (survival focus)

All with a **single unified model** that adapts based on context!
