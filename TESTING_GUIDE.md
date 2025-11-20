# Testing Guide - Model Comparison & Warehouse Application

## Overview

This guide covers testing two trained model architectures and applying them to warehouse scenarios.

## Model Architectures

### Baseline Model (Smaller World View)
- **Observer**: 8 rays × 10 tiles
- **Dimensions**: 92 + 3 context = **95 total**
- **Coverage**: ~25% of 20×20 grid
- **Training script**: `train_context_aware_advanced.py`
- **Typical checkpoint**: `context_aware_advanced_*_policy.pth`

### Expanded Model (Bigger World, Longer View)
- **Observer**: 16 rays × 15 tiles
- **Dimensions**: 180 + 3 context = **183 total**
- **Coverage**: ~60% of 20×20 grid
- **Multi-scale temporal**: Micro (5) + Meso (20) + Macro (50) frames
- **Planning horizon**: 20 steps (vs 5 baseline)
- **Training script**: `train_expanded_faith.py`
- **Typical checkpoint**: `faith_evolution_*_policy.pth`

## Testing Workflow

### Option 1: Automated Complete Testing (Recommended)

```bash
# Run everything automatically
python run_complete_testing.py

# Quick test (20 episodes)
python run_complete_testing.py --quick

# Skip specific tests
python run_complete_testing.py --skip-comparison  # Skip standard games
python run_complete_testing.py --skip-warehouse    # Skip warehouse scenarios

# Specify models manually
python run_complete_testing.py \
    --baseline-model checkpoints/context_aware_advanced_20251118_195410_final_policy.pth \
    --expanded-model checkpoints/faith_evolution_20251120_091144_final_policy.pth \
    --episodes 50
```

This will:
1. Locate the latest trained models automatically
2. Compare them on standard games (Snake, Pac-Man, Dungeon)
3. Test both on warehouse scenarios
4. Generate comprehensive reports

### Option 2: Manual Step-by-Step Testing

#### Step 1: Compare Architectures on Standard Games

```bash
python compare_model_architectures.py \
    --baseline checkpoints/context_aware_advanced_TIMESTAMP_final_policy.pth \
    --expanded checkpoints/faith_evolution_TIMESTAMP_final_policy.pth \
    --episodes 50 \
    --game all
```

**Output**: `model_comparison_report.txt` with detailed performance comparison

#### Step 2: Test on Warehouse Scenarios

```bash
# Test expanded model on all warehouse scenarios
python warehouse_faith_demo.py \
    checkpoints/faith_evolution_TIMESTAMP_final_policy.pth \
    --scenario all \
    --episodes 20

# Test specific scenario
python warehouse_faith_demo.py \
    checkpoints/faith_evolution_TIMESTAMP_final_policy.pth \
    --scenario hidden_shortcut \
    --episodes 50

# Disable faith exploration (pure reactive)
python warehouse_faith_demo.py \
    checkpoints/faith_evolution_TIMESTAMP_final_policy.pth \
    --scenario all \
    --no-faith
```

## Warehouse Scenarios

### 1. Hidden Shortcut
**Challenge**: Discover conditional passageways based on supervisor position
- **Hidden mechanic**: Certain walls become passable when supervisor is far (>5 tiles)
- **Optimal strategy**: Monitor supervisor position, use shortcut when safe
- **Discovery metric**: Shortcut usage count

### 2. Charging Station Dilemma
**Challenge**: Optimal battery management timing
- **Hidden mechanic**: Battery depletes with movement, slowdown at <30% battery
- **Optimal strategy**: Charge at 35-45% battery (proactive vs reactive)
- **Discovery metric**: Optimal charging sessions

### 3. Priority Zone System
**Challenge**: Time-sensitive package prioritization
- **Hidden mechanics**:
  - Red packages: High reward (20) but decay -2 per 10 steps
  - Blue packages: Stable reward (8), no decay
  - Green packages: Medium reward (15) + chain bonus (+5 per consecutive)
- **Optimal strategy**: Red first, chain greens, fill with blues
- **Discovery metric**: Strategy discovery flag

## Interpreting Results

### Architecture Comparison Output

```
Game          Metric         Baseline (92d)    Expanded (180d)   Δ Change
------------------------------------------------------------------------------------
Snake         Avg Score      2.15 ± 1.20       2.50 ± 1.10       +0.35 (+16.3%)
Pacman        Avg Score      3.85 ± 2.40       4.50 ± 2.20       +0.65 (+16.9%)
Dungeon       Avg Score      3.00 ± 1.80       3.80 ± 1.90       +0.80 (+26.7%)
```

**Key metrics to check**:
- **Average Score**: Higher is better
- **Δ Change**: Positive means expanded model outperforms
- **Percentage improvement**: >10% = significant, >20% = major improvement

### Warehouse Demo Output

```
RESULTS:
  Average packages collected: 12.50 ± 3.20
  Max packages: 18
  Average steps: 245.3

ACTION DISTRIBUTION:
  Faith:    25.2% (  1234 actions)
  Planning: 18.5% (   907 actions)
  Reactive: 56.3% (  2759 actions)

HIDDEN MECHANIC DISCOVERIES:
  Total mechanics discovered: 3
    - conditional_shortcut: Walls passable when supervisor distance > 5
      Discovered at step: 142
```

**Success indicators**:
- **High package collection**: >10 per episode is good
- **Mechanic discoveries**: Faith system working if discoveries > 0
- **Action distribution**: Balanced means all systems active

## Decision Matrix

### When to Use Expanded Model

✅ **Use Expanded if**:
- Comparison shows >10% improvement
- Warehouse scenarios show better mechanic discovery
- Planning horizon benefits observed (longer-term strategy)
- Application requires wide field of view

⚠️ **Consider Baseline if**:
- Similar or better performance (expanded needs more training)
- Faster inference needed (smaller model)
- Limited computational resources
- Simpler environments (8 rays sufficient)

## Continue Training

### If Expanded Model Needs More Training

```bash
# Resume from checkpoint
python train_expanded_faith.py \
    --episodes 500 \
    --resume checkpoints/faith_evolution_TIMESTAMP_final_policy.pth \
    --planning-horizon 20 \
    --faith-freq 0.3

# Adjust hyperparameters
python train_expanded_faith.py \
    --episodes 500 \
    --faith-freq 0.4 \           # More exploration
    --planning-freq 0.3 \         # More planning
    --evolution-freq 30           # More frequent evolution
```

### If Baseline Model Needs Improvement

```bash
# Resume baseline training
python train_context_aware_advanced.py \
    --episodes 500 \
    --resume checkpoints/context_aware_advanced_TIMESTAMP_final_policy.pth
```

## Troubleshooting

### Model Not Found
```bash
# List all available checkpoints
ls -lh checkpoints/*final*.pth

# Find latest checkpoints
find checkpoints -name "*final*.pth" -mtime -7  # Last 7 days
```

### Architecture Mismatch Error
```
RuntimeError: size mismatch for q_heads.snake.0.weight:
copying a param with shape torch.Size([64, 183]) from checkpoint,
the shape in current model is torch.Size([64, 95])
```

**Solution**: Use correct observer for model architecture
- Baseline (95 dims) → `TemporalFlowObserver(8, 10)`
- Expanded (183 dims) → `ExpandedTemporalObserver(16, 15)`

The demo scripts auto-detect architecture, but if implementing custom code:

```python
# Auto-detect and create observer
checkpoint = torch.load(policy_path)
state_dict = checkpoint['policy_net']
input_dim = state_dict['q_heads.snake.0.weight'].shape[1]

if input_dim == 95:
    observer = TemporalFlowObserver(num_rays=8, ray_length=10)
elif input_dim == 183:
    observer = ExpandedTemporalObserver(num_rays=16, ray_length=15)
```

### Poor Performance on Warehouse

**Possible causes**:
1. **Need more training**: Model trained on visual games, warehouse is new domain
2. **Faith system needs tuning**: Adjust `--faith-freq` and `--evolution-freq`
3. **Planning disabled**: Ensure world model loaded correctly
4. **Wrong architecture**: Verify observer matches model

**Solutions**:
```bash
# Fine-tune on warehouse domain
python train_expanded_faith.py \
    --episodes 200 \
    --resume checkpoints/faith_evolution_TIMESTAMP_final_policy.pth \
    --env-size 20 \
    --faith-freq 0.4  # More exploration for new domain

# Test with different faith frequencies
for freq in 0.1 0.2 0.3 0.4; do
    python warehouse_faith_demo.py MODEL --faith-freq $freq
done
```

## Next Steps

1. **Run automated testing**:
   ```bash
   python run_complete_testing.py --episodes 50
   ```

2. **Review results**:
   ```bash
   cat model_comparison_report.txt
   ```

3. **Choose model based on performance**

4. **If needed, continue training**:
   ```bash
   python train_expanded_faith.py --episodes 500 --resume BEST_MODEL
   ```

5. **Apply to warehouse**:
   ```bash
   python warehouse_faith_demo.py BEST_MODEL --scenario all --episodes 50
   ```

## Expected Performance Targets

### Standard Games (50 episodes)
- **Snake**: 2.0-3.0 avg score
- **Pac-Man**: 3.5-5.0 avg score
- **Dungeon**: 2.5-4.0 avg score

### Warehouse Scenarios (20 episodes)
- **Hidden Shortcut**: 10-15 packages, 1+ shortcut discoveries
- **Charging Station**: 12-18 packages, 2+ optimal charges
- **Priority Zone**: 10-14 packages, 1+ strategy discoveries

## Files Generated

- `model_comparison_report.txt` - Detailed architecture comparison
- Console output - Warehouse scenario results and discoveries
- `checkpoints/` - Trained model files remain unchanged
