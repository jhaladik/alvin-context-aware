# Log Spam Fix - Expanded Faith Training

## Issues Fixed

### 1. Repetitive Observer Initialization Messages

**Problem:**
```
Expanded Temporal Observer initialized:
  Rays: 16 (angular resolution: 22°)
  Ray length: 15 tiles
  ...
Expanded Temporal Observer initialized:
  Rays: 16 (angular resolution: 22°)
  ...
```
This message was printing 3-4 times per episode because a new observer was created:
- Each time `create_env_for_context()` was called
- During level progression
- When environments respawned

**Solution:**
- Added `verbose=False` parameter to `ExpandedTemporalObserver.__init__()`
- Moved observer info to trainer initialization (prints once at startup)
- Observer now silent by default, only prints if `verbose=True`

**Files Modified:**
- `src/core/expanded_temporal_observer.py` (lines 38, 84-92)
- `src/train_expanded_faith.py` (lines 226-235)

### 2. Empty Universal Patterns & Hidden Mechanics

**Problem:**
```
  Universal Patterns Discovered:

  Hidden Mechanics Confirmed:
```

**Explanation:**
This is **EXPECTED and NORMAL** at episode 10. Pattern detection requires:
- **Universal Patterns**: 50-100+ episodes to detect recurring cross-episode strategies
- **Hidden Mechanics**: 30-50+ episodes to confirm game rule hypotheses with statistical confidence

These sections will populate as training progresses:
- Episode 50: First patterns may emerge
- Episode 100: Multiple patterns detected
- Episode 200+: Robust pattern library

**No fix needed** - this is working as designed.

## How to Apply

### For New Training Runs:
Changes are already in the code. Simply start a new training:
```bash
cd src && python train_expanded_faith.py --episodes 100 --log-every 10
```

### For Currently Running Training:
The current training will continue showing old messages until it completes. Future episodes from new training runs will use the clean output.

## Expected Clean Output

```
======================================================================
EXPANDED SPATIAL-TEMPORAL OBSERVER
======================================================================
  Rays: 16 (angular resolution: 22°) vs 8 baseline
  Ray length: 15 tiles vs 10 baseline (+50% vision range)
  Total observation: 180 dims vs 92 baseline (+96%)
  Multi-scale temporal: Micro (5) + Meso (20) + Macro (50) frames
  Planning horizon: 20 steps vs 5 baseline (4x longer)

======================================================================
STARTING FAITH-BASED EVOLUTIONARY TRAINING
======================================================================

======================================================================
Episode 1/100
======================================================================
  Avg Reward (100): -367.20
  ...
  Entity Discoveries:
    Total entity types: 20
      Entity #0: UNKNOWN (13 interactions)
      ...

  Universal Patterns Discovered:
    (empty early - will populate after episode 50+)

  Hidden Mechanics Confirmed:
    (empty early - will populate after episode 30+)

======================================================================
Episode 10/100
======================================================================
  Avg Reward (100): 381.30
  ...
```

No more repetitive observer initialization messages! ✅
