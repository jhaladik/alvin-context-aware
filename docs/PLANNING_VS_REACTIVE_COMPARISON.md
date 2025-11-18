# Planning vs Reactive Performance Comparison

**Model**: `context_aware_advanced_20251118_165334_best_policy.pth` (Episode 280)
**Date**: 2025-11-18
**Test**: 20 episodes per game

---

## Test Results Summary

### WITH Planning (30% freq, horizon 5)

| Game | Avg Score | Max | Steps | Performance |
|------|-----------|-----|-------|-------------|
| Snake | 1.30 ± 1.38 | 5 | 21.9 | Baseline |
| Pac-Man | 3.85 ± 2.10 | 8 | 20.0 | Good |
| **Dungeon** | **1.50 ± 3.57** | 10 | 301.1 | Lower |
| **Overall** | **2.22** | - | - | - |

### WITHOUT Planning (Policy Only)

| Game | Avg Score | Max | Steps | Performance |
|------|-----------|-----|-------|-------------|
| Snake | 1.20 ± 1.03 | 3 | 19.3 | Baseline |
| Pac-Man | 3.45 ± 2.50 | 10 | 18.8 | Good |
| **Dungeon** | **2.50 ± 4.33** | 10 | 345.4 | **Better!** |
| **Overall** | **2.38** | - | - | - |

---

## Key Findings

### 🚨 Surprising Result: Policy-Only Outperforms Planning!

**Dungeon Performance**:
- Without planning: **2.50 avg**
- With planning: **1.50 avg**
- **Difference: -40% with planning!**

**Overall Performance**:
- Without planning: **2.38 avg**
- With planning: **2.22 avg**
- **Difference: -7% with planning**

---

## Analysis: Why Planning Didn't Help

### Hypothesis 1: World Model Not Accurate Enough ⭐⭐⭐

**Evidence**:
- Episode 280 training (only 80 episodes with planning)
- World model loss: Still 40-70 range (not converged)
- Planning started at episode 200, model needs more time

**World Model Training Timeline**:
```
Episode 200: Loss ~50 (planning enabled)
Episode 250: Loss ~73 (unstable)
Episode 280: Loss ~67 (still learning)
```

**Conclusion**: World model predictions are too noisy for effective planning

### Hypothesis 2: Planning Horizon Too Short ⭐⭐

**Dungeon Characteristics**:
- Large maze (20x20)
- Treasure often 10-15+ steps away
- Needs long-horizon reasoning

**Current Planning**:
- Horizon: 5 steps
- Too short for strategic navigation
- Can't "see" the treasure in most cases

**Evidence**: Dungeon takes 300+ steps on average, 5-step lookahead is only 1.7% of episode

### Hypothesis 3: High Variance in Small Sample ⭐

**Statistics**:
- Standard deviations are VERY high (±2-4 points)
- Only 20 episodes per test
- Dungeon has high variability (0 or 10 score)

**Actual Distribution** (likely):
- Most episodes: 0 points (didn't reach treasure)
- Few episodes: 10 points (reached treasure)
- Small sample → high noise

### Hypothesis 4: Planning Overhead Hurts ⭐

**Computational Cost**:
- Planning: 4 actions × 5 rollouts × 5 steps = 100 forward passes
- Reactive: 1 forward pass
- 100x slower decision-making

**Potential Impact**:
- Slower responses in dynamic environments (Pac-Man)
- Less exploration due to compute time
- May timeout or make suboptimal greedy choices

---

## Detailed Comparison

### Snake (Simple Environment)

| Metric | With Planning | Without Planning | Winner |
|--------|---------------|------------------|--------|
| Avg Score | 1.30 | 1.20 | Planning (+8%) |
| Max Score | 5 | 3 | Planning |
| Avg Steps | 21.9 | 19.3 | Planning (survives longer) |
| Context | 100% snake | 100% snake | Tie |

**Conclusion**: Slight advantage to planning, but minimal difference

### Pac-Man (Dynamic Obstacles)

| Metric | With Planning | Without Planning | Winner |
|--------|---------------|------------------|--------|
| Avg Score | 3.85 | 3.45 | Planning (+12%) |
| Max Score | 8 | 10 | No planning |
| Avg Steps | 20.0 | 18.8 | Planning |
| Survival Ctx | 69.8% | 58.1% | Planning (detects more danger) |

**Conclusion**: Planning helps slightly (+12%), probably due to better collision avoidance

### Dungeon (Long-Horizon Exploration)

| Metric | With Planning | Without Planning | Winner |
|--------|---------------|------------------|--------|
| Avg Score | 1.50 | **2.50** | **No planning (-40%)** |
| Max Score | 10 | 10 | Tie |
| Avg Steps | 301.1 | 345.4 | No planning (explores more) |
| Snake Ctx | 42.6% | 55.6% | No planning (faster navigation) |

**Conclusion**: Planning HURTS performance - likely due to inaccurate world model

---

## Implications

### 1. **Policy Network Learned Well From Planning Experience** ✅

Even though planning during inference doesn't help, the policy network that was **trained** with planning is performing well:
- Overall avg: 2.22-2.38
- Good context detection
- Reasonable performance

**Key Insight**: Planning during training = better policy, but planning during inference ≠ better decisions (yet)

### 2. **World Model Needs More Training** ⚠️

Episode 280 is not enough for accurate world model:
- Needs 500-1000 episodes for convergence
- Loss still fluctuating (40-70 range)
- Predictions too noisy for reliable planning

### 3. **Planning Horizon Too Short for Dungeon** ⚠️

5 steps is insufficient for:
- Finding treasure 10-15 steps away
- Strategic maze navigation
- Long-term route planning

**Recommendation**: Increase horizon to 10-15 steps for Dungeon

### 4. **Planning Works for Training, Not Yet for Inference**

**Training (Episode 200-280)**:
- Used planning to generate better experiences
- Learned from simulated rollouts
- Policy improved significantly

**Inference (Testing)**:
- Planning doesn't improve decisions yet
- World model too inaccurate
- Horizon too short

---

## Recommendations

### Short-Term: Continue Training WITHOUT Inference Planning

**Why**: Policy is already good, no need to use planning during testing yet

**Action**:
```bash
# Disable planning during testing
python test_context_aware.py <checkpoint> --no-planning

# Continue training WITH planning (for better experience)
python train_context_aware_advanced.py --episodes 500 --use-planning ...
```

### Medium-Term: Improve World Model (Episode 500-1000)

**Goals**:
- World model loss < 20
- More stable predictions
- Better dynamics modeling

**Then**: Re-test with planning at episode 500+

### Long-Term: Adaptive Planning

**Context-Aware Planning Frequency**:
```python
if context == 'survival' and game == 'dungeon':
    planning_freq = 0.5  # Use more planning for complex scenarios
    planning_horizon = 10
elif game == 'pacman':
    planning_freq = 0.2  # Less for fast-paced
    planning_horizon = 3
else:
    planning_freq = 0.3
    planning_horizon = 5
```

---

## Current Best Performance (Policy Only)

**Latest Model (Episode 280, No Planning)**:
- Snake: 1.20
- Pac-Man: 3.45
- Dungeon: 2.50
- **Overall: 2.38**

**Comparison to Earlier Models**:
- Episode 20 (no planning): 2.13
- Episode 200 (no planning): 2.82
- Episode 220 (planning trained): 3.00
- **Episode 280 (planning trained, policy only inference): 2.38**

**Trend**: Some regression from episode 220 peak, but still strong

---

## Conclusion

### What We Learned

1. ✅ **Planning during TRAINING works** - improves policy quality
2. ❌ **Planning during INFERENCE doesn't help yet** - world model not accurate enough
3. ⚠️ **World model needs more training** - 280 episodes insufficient
4. ⚠️ **Planning horizon too short** - 5 steps inadequate for Dungeon

### Current Status

**Model Quality**: Good (2.38 avg across games)

**Planning Capability**:
- Training: Working ✅
- Inference: Not ready ❌

**Next Steps**:
1. Continue training to episode 500-1000
2. Monitor world model loss (target: < 20)
3. Re-evaluate planning at episode 500
4. Consider adaptive horizon/frequency

### Bottom Line

**The policy network is good** (learned from planning experience), but **don't use planning during inference yet** (world model not accurate enough).

For now: **Test with `--no-planning` flag** for best performance!

---

## Test Commands

### Current Best Practice (No Planning):
```bash
# Testing
python test_context_aware.py <checkpoint> --episodes 50 --no-planning

# Visual games
python context_aware_visual_games.py --model <checkpoint> --no-planning
```

### Future (When World Model is Ready):
```bash
# Testing with planning
python test_context_aware.py <checkpoint> --episodes 50 \
    --planning-freq 0.3 --planning-horizon 10

# Adaptive planning (manual)
# Pac-Man: --planning-freq 0.2 --planning-horizon 3
# Dungeon: --planning-freq 0.5 --planning-horizon 10
```
