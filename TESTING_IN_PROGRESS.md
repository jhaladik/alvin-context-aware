# Tests Currently Running

**Started**: November 20, 2025, 16:36 UTC

---

## 🔬 Active Test Runs

### 1. Configuration Optimization Test ⏳ RUNNING
**Script**: `run_configuration_tests.py --quick`
**Purpose**: Find optimal configuration (faith freq, planning freq, horizon)
**Status**: In progress

**Testing**:
- Faith frequencies: 0%, 30%
- Planning frequencies: 0%, 20%
- Planning horizon: 20 steps
- Game: Pac-Man
- Episodes per config: 10

**Configurations Being Tested**:
1. F0_P0_H20 (Pure reactive)
2. F0_P20_H20 (Reactive + Planning)
3. F30_P0_H20 (Reactive + Faith)
4. F30_P20_H20 (All systems)

**Expected Output**: Best configuration recommendation with performance scores

---

### 2. Old vs Fixed Model Comparison ⏳ RUNNING
**Script**: `compare_old_vs_fixed.py --quick`
**Purpose**: Direct comparison of bottleneck fix impact
**Status**: In progress

**Models**:
- OLD: `faith_evolution_20251120_091144_final` (500 episodes, standard WM)
- FIXED: `faith_fixed_20251120_162417_final` (700 episodes, fixed WM)

**Testing**:
- Games: Snake, Pac-Man, Dungeon
- Episodes per game: 10
- Planning freq: 20%
- Planning horizon: 20 steps

**Expected Output**: Side-by-side performance comparison, improvement percentage

---

## 📊 Completed Tests

### ✅ Standard Games Test
**Status**: COMPLETED
**Results**:
- Snake: 2.98 avg (Good)
- Pac-Man: **17.62 avg** (EXCELLENT!)
- Dungeon: 0.00 avg (Issue identified)

### ✅ Warehouse Scenarios Test
**Status**: COMPLETED
**Results**:
- Hidden Shortcut: 10.65 avg, 1/1 mechanics discovered ✅
- Charging Station: 4.25 avg, 0/1 mechanics ⚠️
- Priority Zone: 6.30 avg, 3/3 mechanics discovered ✅
- **Overall**: 80% mechanic discovery rate

### ✅ Pac-Man Demo Test
**Status**: COMPLETED
**Results**:
- Average score: 16.40
- Faith actions: 27.9%
- Planning actions: 19.2%
- 18 faith discoveries in 5 episodes

---

## 🎯 What We're Finding Out

### Key Questions Being Answered:

1. **What's the optimal faith frequency?**
   - Testing: 0%, 30%
   - Finding: Will show which exploration level works best

2. **Does planning help?**
   - Testing: With vs without planning
   - Finding: Will show planning contribution to performance

3. **Did the bottleneck fix work?**
   - Testing: Old vs Fixed model head-to-head
   - Finding: Will quantify improvement from architecture fix

4. **What's the best overall configuration?**
   - Testing: All combinations
   - Finding: Will provide definitive recommendation

---

## ⏱️ Estimated Completion

### Configuration Test:
- **Configurations**: 4
- **Episodes per config**: 10
- **Est. time per config**: 3-5 minutes
- **Total ETA**: ~15-20 minutes

### Old vs Fixed Comparison:
- **Games**: 3
- **Episodes per game**: 10 per model
- **Est. time per game**: 3-4 minutes
- **Total ETA**: ~10-15 minutes

**Overall ETA**: Tests should complete within 20-30 minutes

---

## 📁 Output Files

### When Tests Complete:

1. **config_test_results.json** - Detailed configuration test results
2. **config_test_quick.log** - Configuration test log
3. **old_vs_fixed_comparison.log** - Comparison test log

### Analysis Files:

The tests will generate JSON data with:
- Average scores per configuration
- Best configurations per game
- Overall optimal settings
- Statistical comparisons

---

## 🔍 How to Check Progress

### Check Configuration Test:
```bash
cd src
tail -f config_test_quick.log
```

### Check Comparison Test:
```bash
cd src
tail -f old_vs_fixed_comparison.log
```

### Check JSON Results (when complete):
```bash
cd src
cat config_test_results.json | python -m json.tool
```

---

## 📊 What to Expect

### Best Case Scenario:

**Configuration Test** might find:
- Optimal faith freq: 20-30%
- Optimal planning freq: 15-25%
- Planning horizon: 15-20 steps
- Expected improvement: 10-20% over baseline

**Comparison Test** might show:
- FIXED model: +10-20% better than OLD
- Validates bottleneck fix
- Shows planning improvement

### Realistic Scenario:

- Some configurations better for specific games
- FIXED model shows modest improvement
- Clear recommendation on best settings

### Worst Case:

- Configurations perform similarly (still useful - shows robustness)
- FIXED model similar to OLD (architecture cleaner anyway)
- Need more training or tuning

---

## 🚀 Next Steps After Tests Complete

1. **Review Results**:
   - Check which configuration won
   - Verify improvement from fix
   - Analyze game-specific differences

2. **Update Recommendations**:
   - Document optimal settings
   - Update demo scripts with best config
   - Create deployment guide

3. **Final Report**:
   - Comprehensive performance analysis
   - Architecture comparison
   - Production deployment recommendations

---

## 💡 What This Tells Us

These tests will definitively answer:

✅ **Is the fix better?** - Direct A/B comparison
✅ **What settings are best?** - Systematic configuration search
✅ **How much does each component help?** - Ablation study (faith on/off, planning on/off)
✅ **Is the model production-ready?** - Performance validation

---

*Tests started: 2025-11-20 16:36 UTC*
*Expected completion: 2025-11-20 17:00 UTC*
*Status: In Progress...*
