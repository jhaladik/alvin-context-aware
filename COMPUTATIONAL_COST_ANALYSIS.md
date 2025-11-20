# Computational Cost Analysis: Expanded vs Baseline

## TL;DR: 4x Slower Due to Planning Horizon

**Root cause:** Planning horizon increased from 5 to 20 steps = **4x more planning computation**

---

## Detailed Breakdown

### Planning Computation Per Decision

**Monte Carlo Tree Search configuration:**
- Actions to evaluate: 4 (up, down, left, right)
- Rollouts per action: 5
- Planning horizon: **20 steps** (expanded) vs **5 steps** (baseline)

**Forward passes per planning decision:**

| Component | Baseline (5 steps) | Expanded (20 steps) | Multiplier |
|-----------|-------------------|---------------------|------------|
| World model | 4 × 5 × 5 = 100 | 4 × 5 × 20 = 400 | **4x** |
| Policy network | 4 × 5 × 4 = 80 | 4 × 5 × 19 = 380 | **4.75x** |
| **Total** | **180** | **780** | **4.3x** |

---

## Per Episode Cost (125 steps average)

**Planning frequency: 30%**
- Planning decisions: ~37 per episode
- Forward passes per episode:
  - Baseline: 37 × 180 = **6,660**
  - Expanded: 37 × 780 = **28,860**
  - **Ratio: 4.3x slower**

---

## Additional Factors

### 1. Larger Networks
- Input dimension: 183 (expanded) vs 95 (baseline)
- Network size: ~2x more parameters
- **Impact: +30-50% slower per forward pass**

### 2. Complex Observer
- Rays: 16 vs 8 (2x raycasting)
- Multi-scale temporal buffers (micro/meso/macro)
- Pattern detection computations
- **Impact: +20-30% slower per observation**

### 3. Observer Overhead
```python
# ExpandedTemporalObserver additional processing:
- _compute_micro_patterns(): velocity tracking, danger detection
- _compute_meso_patterns(): ghost mode detection (chase/scatter/random)
- _compute_macro_patterns(): strategic trend analysis
```

---

## Total Slowdown Estimate

| Factor | Slowdown |
|--------|----------|
| Planning horizon (20 vs 5) | 4.3x |
| Larger networks (183 vs 95 dims) | 1.4x |
| Complex observer (multi-scale) | 1.2x |
| **Combined** | **~7-8x slower** |

---

## Time Estimates

**Baseline (92-dim, 5-step planning):**
- Per episode: ~2-3 seconds
- 50 episodes: ~2-3 minutes
- 100 episodes: ~4-5 minutes

**Expanded (180-dim, 20-step planning):**
- Per episode: ~15-20 seconds
- 50 episodes: **~15-20 minutes**
- 100 episodes: **~30-40 minutes**

---

## Is This Expected? YES! ✅

The slowdown is **completely normal** and **necessary** for the expanded capability:

1. **20-step planning horizon** required to detect ghost behavior modes (chase/scatter transitions happen over 10-20 steps)
2. **Larger observation space** (180 dims) provides richer spatial-temporal context
3. **Multi-scale temporal processing** captures patterns at different timescales

**Trade-off:**
- **Baseline:** Fast but limited vision (8 rays, 10 tiles, 5-step planning)
- **Expanded:** Slower but superior capability (16 rays, 15 tiles, 20-step planning)

---

## Optimization Options (if needed)

### 1. Reduce Rollouts (Fast Fix)
```python
num_rollouts = 5  # Current
num_rollouts = 3  # Faster (-40% planning time)
```
**Impact:** Slight quality reduction, but still much better than no planning

### 2. Adaptive Planning Horizon
```python
# Use shorter horizon when far from danger, longer when near ghosts
planning_horizon = 10 if entity_distance > 5 else 20
```
**Impact:** ~2x speedup while maintaining quality in critical situations

### 3. Planning Frequency Adjustment
```python
planning_freq = 0.30  # Current
planning_freq = 0.20  # Faster (-33% planning time)
```
**Impact:** Less planning, but faith + reactive can compensate

---

## Recommendation

**Keep current settings** for the initial 50-episode test to get **honest performance measurements**.

If results are excellent (35-50 avg Pac-Man score), then the 7-8x slowdown is **worth it** for the capability gain.

If you need faster testing iterations, try:
```bash
python test_expanded_faith.py <checkpoint> --episodes 20 --planning-freq 0.2
```

This will run ~3x faster while still showing if the expanded approach works.

---

## Current Test Progress

Your 50-episode test (3 games × 50 episodes) estimated time:
- Snake: ~15 minutes
- Pac-Man: ~15 minutes
- Dungeon: ~15 minutes
- **Total: ~45-50 minutes**

vs baseline test: ~6-8 minutes total

**The wait is worth it!** We're testing if expanded spatial-temporal capacity can achieve the target +50-100% improvement.
