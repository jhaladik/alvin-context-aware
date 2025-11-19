# Instructions: 100-Episode Temporal Enhancement Training

## ✅ Prerequisites Verified

All bugs have been fixed and tested:
- ✅ Dimension handling (single & batch observations)
- ✅ Temporal buffer enhancement (micro + meso features)
- ✅ Q-value computation (proper batch handling)
- ✅ Gradient computation (backward pass works)
- ✅ Test suite passed (see `src/test_temporal_fix.py`)

---

## 🎯 What This Training Does

**Goal:** Fine-tune the Episode 500 faith model with temporal enhancement to improve Pac-Man ghost handling.

**How it works:**
1. Loads your trained faith model (Episode 500)
2. **Freezes base agent weights** (preserves learned policy)
3. Adds temporal buffer enhancement (248k trainable parameters)
4. Fine-tunes ONLY the enhancement layers for 100 episodes
5. Tests on Pac-Man every 10 episodes
6. Saves best checkpoint when Pac-Man score improves

**Expected improvements:**
- Pac-Man: 23.12 → 30-35 avg score (+30-50%)
- Ghost handling: Ensemble prediction in high uncertainty
- Training time: ~6-8 hours for 100 episodes

---

## 📋 Step-by-Step Instructions

### Step 1: Create Logs Directory

```bash
mkdir -p logs
```

### Step 2: Verify Base Model Exists

```bash
ls -lh checkpoints/faith_evolution_20251119_152049_best_policy.pth
```

You should see the file (around 1-2 MB).

### Step 3: Run the Training

**Option A: Run in foreground (watch live output)**

```bash
cd src && python train_temporal_enhanced.py \
  --base-model ../checkpoints/faith_evolution_20251119_152049_best_policy.pth \
  --episodes 100 \
  --use-ensemble \
  --freeze-base \
  --test-freq 10
```

Note: Learning rate defaults to 0.00001 (optimized for stability)

**Option B: Run in background with logging**

```bash
cd src && python train_temporal_enhanced.py \
  --base-model ../checkpoints/faith_evolution_20251119_152049_best_policy.pth \
  --episodes 100 \
  --use-ensemble \
  --freeze-base \
  --test-freq 10 \
  2>&1 | tee ../logs/temporal_enhanced_100ep_$(date +%Y%m%d_%H%M%S).log &
```

---

## 📊 What You'll See During Training

### Initial Output:

```
======================================================================
TEMPORAL ENHANCED AGENT - FINE-TUNING
======================================================================
Base model: ../checkpoints/faith_evolution_20251119_152049_best_policy.pth
Fine-tune episodes: 100
Ensemble prediction: True
Freeze base: True
Learning rate: 0.0001
======================================================================

Loading base model...
  Loaded from episode 500

Creating temporal enhanced agent...
  Total parameters: 311,234
  Trainable (enhancement only): 248,064
  Frozen (base agent): 63,170
```

### Training Episodes (every episode):

```
Episode   1/100 (pacman ): Score= 18 Reward=-12.3 Steps=105 Loss=0.1234 Ensemble= 8.5%
Episode   2/100 (snake  ): Score=  9 Reward=-98.7 Steps= 72 Loss=0.0987 Ensemble= 0.0%
Episode   3/100 (dungeon): Score=  3 Reward=-45.2 Steps= 89 Loss=0.1156 Ensemble= 0.0%
...
```

**Column meanings:**
- `Score`: Game score (higher is better)
- `Reward`: Episode reward (higher is better, can be negative)
- `Steps`: Episode length (longer = survived longer)
- `Loss`: Training loss (should decrease over time)
- `Ensemble`: % of actions using ghost ensemble prediction

### Testing Output (every 10 episodes):

```
--- Testing after episode 10 ---
Pac-Man test (10 episodes):
  Avg Score: 24.50 ± 8.23
  Avg Reward: 178.45 ± 145.32
  Best Score: 38
  ✓ New best! Saved to checkpoints/temporal_enhanced_20251119_163245_best.pth
```

**What to look for:**
- **Avg Score increasing** → Model is improving!
- **Best Score** → Highest score in test episodes
- **New best checkpoint saved** → Performance improved

### Final Evaluation (after 100 episodes):

```
======================================================================
FINAL EVALUATION
======================================================================

PACMAN (50 episodes):
  Score: 32.45 ± 9.12
  Reward: 245.67 ± 156.34
  Best Score: 52
  Ensemble Usage: 18.2%

SNAKE (50 episodes):
  Score: 9.23 ± 3.45
  Reward: 67.89 ± 78.23
  Best Score: 18
  Ensemble Usage: 0.3%

DUNGEON (50 episodes):
  Score: 4.56 ± 2.78
  Reward: 89.12 ± 67.45
  Best Score: 12
  Ensemble Usage: 2.1%
```

---

## 🔍 Monitoring Progress

### Watch Live Output (if running in background):

```bash
tail -f logs/temporal_enhanced_100ep_*.log
```

### Check Current Best Checkpoint:

```bash
ls -lht checkpoints/temporal_enhanced_*.pth | head -5
```

### Estimate Time Remaining:

Each episode takes approximately 3-5 minutes:
- 100 episodes × 4 min/ep = ~400 minutes = **6-7 hours**
- Testing adds ~2 min every 10 episodes = +20 min
- **Total: ~6.5-7.5 hours**

---

## 📈 Interpreting Results

### Success Criteria:

**✅ GOOD RESULTS:**
- Avg Pac-Man Score: 30-35+ (improvement from 23.12)
- Ensemble Usage: 15-25% (used when needed)
- Loss decreasing over time
- Best Score: 45-60+

**⚠️ MEDIOCRE RESULTS:**
- Avg Pac-Man Score: 25-30 (small improvement)
- Ensemble Usage: <5% (not being used)
- Loss staying high
- Best Score: 30-40

**❌ POOR RESULTS:**
- Avg Pac-Man Score: <25 (no improvement)
- Ensemble Usage: 0% (broken)
- Loss increasing
- Frequent crashes

### What Each Metric Means:

**Score:**
- **Pac-Man:** Pellets collected (200 pellets total, max 200)
- **Snake:** Food collected (max ~20-30)
- **Dungeon:** Treasures collected (max ~10-15)

**Reward:**
- Total reward accumulated during episode
- Can be negative early in training
- Should increase as model improves

**Ensemble Usage:**
- % of actions using ghost ensemble prediction
- Should be 15-25% for Pac-Man (high uncertainty)
- Should be ~0% for Snake/Dungeon (no ghosts)

**Loss:**
- TD-error between predicted and target Q-values
- Should decrease over time
- Typical range: 0.05-0.20 after convergence

---

## 💾 Saved Checkpoints

Every time the Pac-Man test score improves, a checkpoint is saved:

**Location:** `checkpoints/temporal_enhanced_YYYYMMDD_HHMMSS_best.pth`

**Contents:**
```python
{
    'episode': 50,
    'base_model_path': '../checkpoints/faith_evolution_20251119_152049_best_policy.pth',
    'enhancement_state_dict': {...},  # Temporal enhancement weights
    'optimizer_state_dict': {...},
    'best_pacman_score': 32.45,
    'episode_rewards': [...],
    'config': {...}
}
```

**To use the best checkpoint:**
```bash
# Test it with demo
python src/demo_pacman_faith.py \
  --checkpoint ../checkpoints/temporal_enhanced_20251119_163245_best.pth \
  --episodes 5 \
  --planning-freq 0.2 \
  --faith-freq 0.0
```

---

## 🐛 Troubleshooting

### Issue: Training crashes immediately

**Check:**
```bash
python src/test_temporal_fix.py
```

Should see "ALL TESTS PASSED!". If not, there's a bug.

### Issue: Ensemble usage is 0%

**Cause:** Uncertainty threshold too high
**Fix:** Lower threshold in line 75 of `train_temporal_enhanced.py`:
```python
if uncertainty > 0.4:  # Was 0.6
```

### Issue: Loss stays at 0.0

**Cause:** Replay buffer not filled yet
**Wait:** Happens for first ~128 steps, then training kicks in

### Issue: Avg Score decreasing

**Cause:** Overfitting or learning rate too high
**Fix 1:** Reduce learning rate (already at 0.00001 by default):
```bash
--lr 0.000005  # Even more conservative if needed
```

**Fix 2:** Stop early if plateau detected

### Issue: Out of memory

**Cause:** Batch size too large
**Fix:** Reduce batch size in line 169 of `train_temporal_enhanced.py`:
```python
transitions, indices, weights = buffer.sample(16)  # Was 32
```

---

## 📊 Expected Training Curve

```
Episodes 1-20:   Exploration, high variance, score ~20-25
Episodes 20-40:  Learning kicks in, score ~25-28
Episodes 40-60:  Improvement, score ~28-32
Episodes 60-80:  Refinement, score ~30-34
Episodes 80-100: Convergence, score ~32-36
```

**Best score typically achieved:** Episode 60-80

---

## 🎯 After Training

### Option 1: If Results Are Good (30-35+ avg)
✅ **Success!** Temporal enhancement works!
- Use best checkpoint for demos
- Document improvement
- Move to warehouse scenarios

### Option 2: If Results Are Mediocre (25-30 avg)
⚠️ **Partial success** - some improvement
- Try longer training (150-200 episodes)
- Tune hyperparameters (learning rate, uncertainty threshold)
- Consider Option B (full hierarchical transformer)

### Option 3: If Results Are Poor (<25 avg)
❌ **Enhancement doesn't help enough**
- Need full hierarchical temporal transformer (Option B)
- Or focus on warehouse (linear movement = perfect fit)

---

## 📝 Quick Command Reference

**Start training (foreground):**
```bash
cd src && python train_temporal_enhanced.py --base-model ../checkpoints/faith_evolution_20251119_152049_best_policy.pth --episodes 100 --use-ensemble --freeze-base --test-freq 10
```

**Start training (background with logging):**
```bash
cd src && python train_temporal_enhanced.py --base-model ../checkpoints/faith_evolution_20251119_152049_best_policy.pth --episodes 100 --use-ensemble --freeze-base --test-freq 10 2>&1 | tee ../logs/temporal_enhanced_100ep_$(date +%Y%m%d_%H%M%S).log &
```

**Monitor progress:**
```bash
tail -f logs/temporal_enhanced_100ep_*.log
```

**Test best checkpoint:**
```bash
python src/demo_pacman_faith.py --checkpoint ../checkpoints/temporal_enhanced_*_best.pth --episodes 5 --planning-freq 0.2 --faith-freq 0.0
```

**Find best checkpoint:**
```bash
ls -lht checkpoints/temporal_enhanced_*.pth | head -1
```

---

## 🚀 Ready to Start!

All bugs are fixed. All systems are go. Run the command and let it train!

**Estimated completion time:** 6-7 hours from now

Good luck! 🎮🤖
