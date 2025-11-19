# Option A: Expanded Spatial-Temporal Capacity - Implementation Plan

## ✅ Phase 1: Expanded Temporal Observer (COMPLETE)

**File:** `src/core/expanded_temporal_observer.py`

**What We Built:**

### Spatial Expansion (2x Coverage)
- **Rays:** 16 vs 8 (22° angular resolution vs 45°)
- **Range:** 15 tiles vs 10 (+50% vision distance)
- **Coverage:** ~60% of grid vs ~25%
- **Benefit:** See ghosts approaching from farther away

### Multi-Scale Temporal Understanding
```
Micro buffer (5 frames):   Immediate threats, collision avoidance
Meso buffer (20 frames):   Tactical patterns, ghost behavior modes
Macro buffer (50 frames):  Strategic trends, long-term survival
```

### Advanced Pattern Features (32 dims)

**Micro Patterns (8 dims):**
- Danger oscillation
- Movement consistency
- Trap detection
- Escape success rate
- Entity stability
- Reward availability
- Collision risk

**Meso Patterns (8 dims):**
- Ghost behavior modes (chase/scatter/random detection!)
- Zone coverage
- Tactical quality
- Survival stability
- Reward collection rate
- Evasion skill

**Macro Patterns (8 dims):**
- Strategic progress
- Exploration coverage
- Long-term survival quality
- Danger exposure time
- Cycling behavior
- Learning signal
- Efficiency
- Stamina

**Cross-Scale Patterns (8 dims):**
- Micro-meso alignment
- Meso-macro alignment
- Regime change detection
- Uncertainty quantification
- Prediction confidence
- Adaptation speed
- Strategic coherence
- Risk level

### Total Observation: 180 dims
```
Current features:     80 dims (16 rays + global info)
Delta features:       68 dims (immediate changes)
Multi-scale temporal: 32 dims (pattern features)
Total:               180 dims (+96% vs 92 dim baseline)
```

---

## 🚧 Phase 2: Modified Training Script (NEXT)

### Changes Required:

**1. Use Expanded Observer**
```python
from core.expanded_temporal_observer import ExpandedTemporalObserver

# Replace:
observer = TemporalFlowObserver(num_rays=8, ray_length=10)  # 92 dims

# With:
observer = ExpandedTemporalObserver(num_rays=16, ray_length=15)  # 180 dims
```

**2. Update Network Input Dimensions**
```python
# Old:
obs_dim = 92  # Temporal observer
obs_with_context = 95  # 92 + 3 context features

# New:
obs_dim = 180  # Expanded temporal observer
obs_with_context = 183  # 180 + 3 context features
```

**3. Extend Planning Horizon**
```python
# Old:
planning_horizon = 5  # Only 4% of 122-step average game

# New:
planning_horizon = 20  # 16% of game - can see ghost mode patterns!

# Or adaptive:
if min_ghost_distance < 8:
    planning_horizon = 30  # Danger - plan carefully
else:
    planning_horizon = 15  # Safe - quicker planning
```

**4. Create New Training Script**
```
File: src/train_expanded_faith.py
Based on: src/train_with_faith.py
Modifications:
- ExpandedTemporalObserver instead of TemporalFlowObserver
- obs_dim=183 (180 + 3 context)
- planning_horizon=20
- Same faith system, continuous motivation, world model
```

---

## 📊 Expected Improvements

### Current Baseline (Episode 500):
```
Pac-Man: 23.12 avg score (29.2% completion)
- Observation: 92 dims (8 rays × 10 length)
- Planning: 5 steps (4% of game)
- Problem: Can't see ghosts until close, can't predict behavior modes
```

### Expected with Option A:
```
Pac-Man: 35-50 avg score (45-65% completion)
- Observation: 180 dims (16 rays × 15 length + multi-scale)
- Planning: 20 steps (16% of game)
- Advantages:
  ✓ See ghosts approaching earlier (15 vs 10 tiles)
  ✓ Detect ghost behavior modes (chase/scatter patterns)
  ✓ Plan 20 steps ahead (enough to see mode patterns)
  ✓ Strategic evasion (not just reactive dodging)
```

**Improvement:** +50-100% performance boost

---

## 🎯 Key Insight Addressed

**Your insight:** "Ghost decisions are atomic (instant), but we need to expand vision and planning to see the bigger picture."

**How Option A delivers:**

1. **Expanded Vision (Spatial):**
   - See 60% of grid vs 25%
   - Catch ghost approaches early
   - More time to plan evasion

2. **Longer Planning (Temporal):**
   - 20 steps vs 5
   - See behavior patterns emerge (10-20 step patterns)
   - Recognize modes: "Ghost chased me 15 times in last 20 steps → CHASE mode"

3. **Multi-Scale Understanding:**
   - Micro: "Ghost approaching now" (immediate)
   - Meso: "Ghost in CHASE mode" (tactical, 20 steps)
   - Macro: "Overall danger decreasing" (strategic, 50 steps)

4. **Pattern Recognition:**
   - Ghost behavior modes detected automatically
   - No need to sub-sample (you were right - atomic decisions!)
   - Instead: recognize patterns over longer timescales

---

## 🔧 Implementation Steps

### Step 1: Create Training Script (30 min)
```bash
# Copy and modify existing faith training
cp src/train_with_faith.py src/train_expanded_faith.py

# Modifications needed:
1. Import ExpandedTemporalObserver
2. Change obs_dim to 183
3. Set planning_horizon=20
4. Test with 1 episode to verify dimensions
```

### Step 2: Quick Dimension Test (5 min)
```bash
# Verify network accepts 183-dim input
python src/train_expanded_faith.py --episodes 1 --test
```

### Step 3: Full Training (10 hours)
```bash
# Train for 500 episodes like original faith training
python src/train_expanded_faith.py \
  --episodes 500 \
  --planning-freq 0.2 \
  --planning-horizon 20 \
  --faith-freq 0.05
```

### Step 4: Evaluation (30 min)
```bash
# Test on Pac-Man with expanded observer
python src/test_faith.py \
  checkpoints/expanded_faith_best_policy.pth \
  --episodes 50 \
  --planning-freq 0.2 \
  --planning-horizon 20
```

---

## 📈 Success Criteria

**Minimum Success:**
- Pac-Man: 30+ avg score (vs 23.12 baseline)
- +30% improvement
- Ghost behavior modes visible in logs

**Target Success:**
- Pac-Man: 40+ avg score
- +75% improvement
- Clear evidence of strategic evasion (not just reactive)

**Exceptional Success:**
- Pac-Man: 50+ avg score
- +115% improvement
- Approaching DQN SOTA (~60 avg)

---

## 🆚 Comparison to Failed Approach

### Temporal Enhancement (Failed):
- ❌ Froze base agent (couldn't adapt to enhanced obs)
- ❌ 95% performance regression
- ❌ Ensemble never used (0%)
- **Result:** 1.16 avg score (worse than random!)

### Option A (New Approach):
- ✅ Train from scratch with expanded obs
- ✅ Agent learns to use multi-scale features
- ✅ Network designed for 180-dim input from start
- ✅ No architectural mismatch
- **Expected:** 35-50 avg score (+50-100%)

**Key difference:** Not trying to retrofit - building properly from scratch!

---

## ⏱️ Timeline

**Total time:** ~11 hours

- ✅ Phase 1 (Expanded Observer): 2 hours - **COMPLETE**
- 🚧 Phase 2 (Training Script): 30 min - **NEXT**
- ⏳ Phase 3 (Training): 10 hours
- ⏳ Phase 4 (Evaluation): 30 min

---

## 🚀 Next Steps

1. **Create `train_expanded_faith.py`** (You decide when to proceed)
2. **Test with 1 episode** (verify dimensions)
3. **Launch 500-episode training** (overnight run)
4. **Evaluate results** (morning after)
5. **Compare to baseline** (23.12 avg)

---

## 💡 Why This Will Work

**The fundamental principle:**

> "Ghost decisions are atomic, but behavior patterns emerge over 10-20 steps.
> By planning 20 steps ahead with expanded vision, we can recognize chase/scatter
> modes and evade strategically instead of reacting blindly."

**Technical implementation:**
- Spatial expansion: See threats earlier
- Temporal expansion: Recognize patterns
- Multi-scale features: Understand at all timescales
- Longer planning: Act on strategic understanding

**No architectural mismatch:**
- Network trained with 180-dim obs from episode 1
- No freezing, no retrofitting
- Clean, purposeful design

This is **fundamentally sound** and should deliver the improvement we're targeting!

---

Ready to proceed with Phase 2 (create training script)?
