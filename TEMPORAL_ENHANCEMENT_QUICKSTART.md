# Temporal Buffer Enhancement - Quick Fix for Ghost Prediction

## What Is This?

A **quick fix** (tonight, 2-3 hours) to improve Pac-Man ghost handling by adding:
1. **Multi-scale temporal buffers** (micro 5-frame + meso 50-frame)
2. **Uncertainty quantification** (know when to trust temporal predictions)
3. **Ensemble ghost prediction** (optimistic/expected/pessimistic scenarios)
4. **Safe zone computation** (find positions safe in ALL scenarios)

**No full retrain needed** - just fine-tune existing model for 50 episodes!

---

## Files Created

### 1. `src/core/temporal_buffer_enhancement.py`
**Purpose**: Adds temporal buffering to existing agent

**Key Classes:**
- `TemporalBufferEnhancement`: Multi-scale temporal features (micro + meso)
- `GhostEnsemblePredictor`: Predicts ghosts using 3 scenarios

**Features:**
- Micro buffer: Last 5 frames (high resolution, immediate threats)
- Meso buffer: Last 50 frames (patterns, choreography)
- Uncertainty estimation: Know when temporal predictions are unreliable
- Ensemble prediction: 3 ghost position hypotheses (best/expected/worst case)
- Safe zones: Positions safe in ALL scenarios

### 2. `src/train_temporal_enhanced.py`
**Purpose**: Fine-tune existing model with temporal enhancement

**What it does:**
- Loads your trained faith model (Episode 500)
- Wraps it with TemporalBufferEnhancement
- **Freezes base agent** (only trains enhancement layers)
- Fine-tunes for 50 episodes (2-3 hours)
- Tests every 10 episodes on Pac-Man
- Saves best checkpoint

**Parameters:**
- Only **248k trainable** parameters (out of 311k total)
- Base agent frozen → preserves learned policy
- Enhancement layers learn temporal patterns

---

## Usage

### Quick Start (Tonight):

```bash
# From repository root
python src/train_temporal_enhanced.py \
  --base-model checkpoints/faith_evolution_20251119_152049_best_policy.pth \
  --episodes 50 \
  --use-ensemble \
  --test-freq 10
```

**What happens:**
1. Loads Episode 500 model
2. Adds temporal enhancement (248k params)
3. Fine-tunes for 50 episodes (~2-3 hours)
4. Tests on Pac-Man every 10 episodes
5. Saves best checkpoint

**Expected output:**
```
Episode   1/50 (pacman ): Score= 12 Reward=-45.2 Steps= 87 Loss=0.1234 Ensemble= 12.5%
Episode   2/50 (snake  ): Score=  8 Reward=-120.5 Steps= 65 Loss=0.0987 Ensemble= 0.0%
...
--- Testing after episode 10 ---
Pac-Man test (10 episodes):
  Avg Score: 24.50 ± 8.23
  Avg Reward: 178.45 ± 145.32
  Best Score: 38
  ✓ New best! Saved to checkpoints/temporal_enhanced_20251119_163245_best.pth
...
```

### Full Training (50 episodes):

```bash
python src/train_temporal_enhanced.py \
  --base-model checkpoints/faith_evolution_20251119_152049_best_policy.pth \
  --episodes 50 \
  --use-ensemble \
  --freeze-base \
  --lr 0.0001 \
  --test-freq 10
```

**Arguments:**
- `--base-model`: Your trained model checkpoint
- `--episodes`: Number of fine-tuning episodes (default: 50)
- `--use-ensemble`: Enable ghost ensemble prediction (recommended)
- `--freeze-base`: Freeze base agent weights (default: True)
- `--lr`: Learning rate for enhancement layers (default: 0.0001)
- `--test-freq`: Test every N episodes (default: 10)

---

## How It Works

### 1. Temporal Buffering

**Before (TemporalFlowObserver):**
```python
observation = [current_features, delta(t-1 to t)]
# Only 1-frame delta!
```

**After (TemporalBufferEnhancement):**
```python
observation = [
    current_features,          # What I see NOW
    micro_features (5 frames), # Immediate patterns
    meso_features (50 frames), # Choreography
    uncertainty                # How reliable are predictions?
]
```

### 2. Ensemble Ghost Prediction

Instead of single linear prediction:
```python
# Old: ghost_pos(t+5) = ghost_pos(t) + velocity * 5
# Fails because ghosts don't move linearly!
```

We use **3 scenarios**:
```python
predictions = {
    'optimistic': ghost_moves_away(agent_pos),  # Best case
    'expected': mean_velocity_prediction(),      # Most likely
    'pessimistic': ghost_chases_agent(),         # Worst case
}

# Find positions safe in ALL scenarios
safe_zones = positions_safe_in_all(predictions)

# Move toward nearest safe zone + reward
action = move_to_safe_zone_near_reward(safe_zones, reward_pos)
```

### 3. Uncertainty-Based Action Selection

```python
if uncertainty > 0.6:  # High uncertainty
    # Use ensemble prediction (safer)
    action = ensemble_based_action(ghost_predictions, agent_pos, reward_pos)
else:  # Low uncertainty
    # Use normal Q-values
    action = q_value_based_action(enhanced_obs)
```

---

## Expected Performance

### Current (Episode 500 Base Model):
```
Pac-Man: 23.12 avg score, 29.2% completion
  Faith:    0% (testing)
  Planning: 20%
  Reactive: 80%
```

### After Temporal Enhancement (Expected):
```
Pac-Man: 30-35 avg score, 38-45% completion  (+30-50% improvement!)
  Ensemble: 15-20% (high uncertainty situations)
  Planning: 15%
  Reactive: 65-70%
```

**Why improvement:**
- Multi-scale temporal understanding (5 + 50 frames)
- Ghost ensemble prediction (handles uncertainty)
- Safe zone computation (avoids traps)
- Uncertainty-based action selection (know when to trust predictions)

---

## Monitoring Training

### Check Progress:
```bash
# Watch live output
tail -f logs/temporal_enhanced_training.log

# Or check background job
python -c "import subprocess; subprocess.run(['tail', '-20', 'logs/temporal_enhanced_training.log'])"
```

### Look for:
1. **Loss decreasing**: Enhancement layers learning
2. **Ensemble usage increasing**: Agent learning when to use ensemble
3. **Test scores improving**: Every 10 episodes
4. **Best checkpoints saved**: When new best Pac-Man score achieved

---

## Troubleshooting

### Issue: Ensemble usage 0%
**Cause**: Uncertainty threshold too high (> 0.6)
**Fix**: Lower threshold in `temporal_buffer_enhancement.py` line 70:
```python
if uncertainty > 0.4:  # Was 0.6
```

### Issue: Loss stays at 0
**Cause**: Not enough samples in replay buffer
**Fix**: Train more episodes before sampling kicks in (happens after ~128 samples)

### Issue: Performance worse
**Cause**: Ensemble predictions too conservative
**Fix**: Adjust safe zone distance in `temporal_buffer_enhancement.py` line 227:
```python
if dist < 2:  # Was 3 - less conservative
```

---

## What's Next?

### If This Works (35-40% completion):
✅ **Success!** Temporal enhancement solves ghost problem
- Commit and document
- Use for demos
- Move to warehouse scenarios

### If Still Struggling (< 35% completion):
→ Need **full hierarchical temporal transformer** (Option B)
- 500-frame macro buffer
- FFT frequency analysis
- Attention mechanism
- Chaos crystallization
- **~10 hours** full retrain

---

## Comparison to Full Solution

### Option A: Temporal Buffer Enhancement (This)
**Time**: 2-3 hours
**Complexity**: Low (wrapper around existing model)
**Expected**: 30-35 avg score
**Best for**: Quick validation tonight

### Option B: Hierarchical Temporal Transformer
**Time**: ~10 hours
**Complexity**: High (full architecture redesign)
**Expected**: 40-50 avg score
**Best for**: Academic paper / maximum performance

---

## Key Insight

**The problem isn't training time** - it's temporal architecture!

Ghosts have **complex choreography** requiring:
- Multiple temporal scales (micro/meso/macro)
- Uncertainty quantification
- Ensemble predictions
- Dynamic focus (zoom in on threats)

This enhancement adds the missing pieces **without** full retrain.

---

## Recommendation

**Try tonight:**
1. Run the 50-episode fine-tuning (started already!)
2. Check results in 2-3 hours
3. If >35% completion → Success!
4. If <35% → Consider Option B (full hierarchical)

**Then decide:**
- Commercial focus → Move to warehouse (current system perfect for linear movement)
- Academic focus → Implement full hierarchical temporal transformer

**Remember:** Warehouse AGVs move linearly → current TemporalFlowObserver is already perfect for that use case!
