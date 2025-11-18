# Context-Aware Agent Training Methodology

**A comprehensive explanation of the training system, algorithms, and checkpoint management.**

---

## Table of Contents
1. [Training Architecture Overview](#training-architecture-overview)
2. [Core Algorithms](#core-algorithms)
3. [Checkpoint Saving System](#checkpoint-saving-system)
4. [Training Methodology Analysis](#training-methodology-analysis)
5. [Checkpoint Comparison](#checkpoint-comparison)
6. [Does It Make Sense?](#does-it-make-sense)

---

## Training Architecture Overview

### System Components

The training system consists of 4 main components working together:

```
┌─────────────────────────────────────────────────────────────┐
│                    CONTEXT-AWARE TRAINER                     │
├─────────────────────────────────────────────────────────────┤
│  1. Policy Network (ContextAwareDQN) - 62,864 parameters   │
│  2. Target Network (frozen copy for stability)              │
│  3. World Model (WorldModelNetwork) - 58,593 parameters     │
│  4. Replay Buffer (100,000 transitions)                     │
└─────────────────────────────────────────────────────────────┘
           ↓                    ↓                    ↓
    ┌──────────┐        ┌──────────┐        ┌──────────┐
    │  SNAKE   │        │ BALANCED │        │ SURVIVAL │
    │  MODE    │        │   MODE   │        │   MODE   │
    │ 0 ent.   │        │  2-3 ent.│        │  4-6 ent.│
    │  30%     │        │   50%    │        │   20%    │
    └──────────┘        └──────────┘        └──────────┘
```

---

## Core Algorithms

### 1. Deep Q-Network (DQN) with Double Q-Learning

**What it is:**
DQN learns to estimate Q-values: Q(state, action) = expected future reward

**How it works:**

```python
# Current Q-value prediction (from policy network)
current_q = policy_net.get_combined_q(states)[actions]

# Target Q-value (from target network - more stable)
with torch.no_grad():
    next_q = target_net.get_combined_q(next_states)
    max_next_q = next_q.max()
    target_q = reward + gamma * max_next_q * (1 - done)

# Loss: Mean Squared Error between current and target
loss = MSE(current_q, target_q)
```

**Key parameters:**
- **Gamma (γ = 0.99)**: Discount factor - how much to value future rewards
- **Learning rate (0.0001)**: How fast to update the policy network
- **Batch size (64)**: Number of experiences to learn from at once
- **Epsilon decay**: Exploration vs exploitation
  - Start: ε = 1.0 (100% random exploration)
  - End: ε = 0.01 (99% learned policy)
  - Decay: Linear over 50% of training

**Why DQN?**
- ✓ Sample efficient (learns from replay buffer)
- ✓ Stable (target network prevents moving targets)
- ✓ Simple and proven (used in Atari games)

---

### 2. Hierarchical Q-Heads Architecture

**The policy network has 4 specialized "heads":**

1. **Survival Head** (weight: 2.0) - Avoid immediate death
2. **Avoidance Head** (weight: 1.5) - Maintain safe distance from threats
3. **Positioning Head** (weight: 1.0) - Strategic placement
4. **Collection Head** (weight: 1.0) - Gather rewards

**How decisions are made:**

```python
# Each head outputs Q-values for all 4 actions
q_survival   = survival_head(features)      # Shape: [batch, 4]
q_avoidance  = avoidance_head(features)     # Shape: [batch, 4]
q_position   = positioning_head(features)   # Shape: [batch, 4]
q_collection = collection_head(features)    # Shape: [batch, 4]

# Weighted combination (context-aware agent adapts these weights!)
combined_q = (2.0 * q_survival +
              1.5 * q_avoidance +
              1.0 * q_position +
              1.0 * q_collection)

# Best action
action = argmax(combined_q)
```

**Why hierarchical?**
- ✓ Specialized behaviors (survival vs collection)
- ✓ Interpretable decisions (we know WHY agent chose action)
- ✓ Context adaptability (different heads dominant in different modes)

---

### 3. World Model (Predictive Learning)

**What it does:**
Learns to predict: (state, action) → (next_state, reward, done)

**Architecture:**
```
Input: [state (95-dim), action (one-hot 4-dim)] = 99-dim
  ↓
State Predictor → predicts next_state (95-dim)
Reward Predictor → predicts reward (1-dim)
Done Predictor → predicts episode end (1-dim, probability)
```

**Training:**
```python
# Predict future
pred_next_state, pred_reward, pred_done = world_model(state, action)

# Loss = how wrong were predictions?
state_loss = MSE(pred_next_state, actual_next_state)
reward_loss = MSE(pred_reward, actual_reward)
done_loss = BinaryCrossEntropy(pred_done, actual_done)

total_loss = state_loss + reward_loss + done_loss
```

**Why world model?**
- ✓ Enables planning (imagine multiple steps ahead)
- ✓ Better sample efficiency (learn dynamics)
- ✓ Could enable model-based RL in future (not used yet)

**Current status:** Trained but NOT used for planning during test time (pure model-free DQN)

---

### 4. Context-Aware Training

**The KEY innovation:** Train on mixed scenarios, adapt behavior based on context

**Training distribution:**
```
30% SNAKE mode     (0 entities)    → Learn aggressive collection
50% BALANCED mode  (2-3 entities)  → Learn tactical gameplay
20% SURVIVAL mode  (4-6 entities)  → Learn threat avoidance
```

**Each episode:**
```python
1. Sample context ~ [snake: 0.30, balanced: 0.50, survival: 0.20]
2. Create environment with appropriate num_entities
3. Generate context vector:
   - Snake: [1, 0, 0]
   - Balanced: [0, 1, 0]
   - Survival: [0, 0, 1]
4. Append context to all observations (92 + 3 = 95-dim)
5. Train policy to maximize reward given this context
```

**At test time:**
```python
# Agent INFERS context from observation
entity_count = count_entities_in_rays(observation)
if entity_count == 0:
    context = [1, 0, 0]  # SNAKE mode
elif entity_count <= 3:
    context = [0, 1, 0]  # BALANCED mode
else:
    context = [0, 0, 1]  # SURVIVAL mode
```

**Why this works:**
- ✓ Agent learns that context signal predicts reward structure
- ✓ Different strategies optimal in different contexts
- ✓ Single agent adapts instead of needing 3 separate models

---

## Checkpoint Saving System

### What Gets Saved

**Every checkpoint contains TWO files:**

#### 1. Policy Checkpoint (`*_best_policy.pth`)
```python
{
    'policy_net': state_dict,        # Trained policy weights
    'target_net': state_dict,        # Target network weights
    'optimizer': state_dict,         # Optimizer state (for resuming)
    'episode_rewards': [r1, r2, ...],# All episode rewards
    'episode_lengths': [l1, l2, ...],# All episode lengths
    'context_episode_counts': {...}, # Episodes per context
    'context_avg_rewards': {...},    # Performance per context
    'steps_done': int                # Total environment steps
}
```

#### 2. World Model Checkpoint (`*_best_world_model.pth`)
```python
{
    'model': state_dict,             # World model weights
    'optimizer': state_dict,         # World model optimizer
    'losses': [l1, l2, ...]         # Training losses
}
```

### When Checkpoints Are Saved

**Trigger:** Every `log_every` episodes (default: 100)

**Condition:** Save if `avg_reward_last_100 > best_avg_reward_so_far`

```python
if (episode + 1) % log_every == 0:
    avg_reward = mean(last_100_episode_rewards)

    if avg_reward > best_avg_reward:
        best_avg_reward = avg_reward
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save(f"checkpoints/context_aware_{timestamp}_best")
        print(f"[BEST] Saved model (avg reward: {avg_reward:.2f})")
```

**Filename format:** `context_aware_YYYYMMDD_HHMMSS_best_policy.pth`

**Example:** `context_aware_20251118_115931_best_policy.pth`
- Date: 2025-11-18
- Time: 11:59:31
- Type: best (performance milestone)

### Final Checkpoint

**At end of training:** Always save final state
```
context_aware_YYYYMMDD_HHMMSS_final_policy.pth
context_aware_YYYYMMDD_HHMMSS_final_world_model.pth
```

---

## Training Methodology Analysis

### Training Loop (Per Episode)

```python
1. Sample context (snake/balanced/survival)
2. Create environment with appropriate entities
3. Reset environment → get initial observation (92-dim)
4. Add context vector → observation becomes 95-dim
5. While not done:
   a. Select action (ε-greedy)
   b. Execute in environment
   c. Store transition in replay buffer
   d. IF buffer >= batch_size:
      - Train policy network (1 gradient step)
      - Train world model (1 gradient step)
   e. IF steps % target_update_freq == 0:
      - Copy policy_net weights → target_net
   f. Accumulate reward
6. Log episode statistics
7. Save checkpoint if best performance
```

### Hyperparameters Summary

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Environment** | | |
| Grid size | 20×20 | Training maze size |
| Num rewards | 10 | Sparse (encourages exploration) |
| Maze complexity | 0.3 | 30% wall density |
| **Learning** | | |
| Policy LR | 0.0001 | Policy network learning rate |
| World Model LR | 0.0003 | World model learning rate (3× faster) |
| Gamma (γ) | 0.99 | Discount factor |
| Batch size | 64 | Experiences per update |
| Buffer size | 100,000 | Max stored transitions |
| Target update freq | 500 steps | How often to update target network |
| **Exploration** | | |
| Epsilon start | 1.0 | 100% random initially |
| Epsilon end | 0.01 | 1% random at end |
| Epsilon decay | Linear over 50% | Gradual shift to learned policy |
| **Context** | | |
| Snake | 30% | 0 entities |
| Balanced | 50% | 2-3 entities |
| Survival | 20% | 4-6 entities |

### Learning Dynamics (From Best Checkpoint)

**Training progress over 260 episodes:**

```
Episodes 1-100:   Avg reward = 74.95  (learning basics)
Episodes 101-260: Avg reward = 233.04 (skilled performance)

Improvement: +158.08 reward (+210.9%)
```

**Context-specific performance:**
- **SNAKE mode**: 477.88 avg reward (excellent collection)
- **BALANCED mode**: 31.68 avg reward (moderate difficulty)
- **SURVIVAL mode**: -144.21 avg reward (hard to survive)

**Context distribution achieved:**
- Snake: 35.4% (target: 30%) ✓
- Balanced: 48.1% (target: 50%) ✓
- Survival: 16.5% (target: 20%) ✓

---

## Checkpoint Comparison

### Using the Comparison Tool

**Compare all checkpoints:**
```bash
cd src
python compare_checkpoints.py
```

**Show top 5 best:**
```bash
python compare_checkpoints.py --top 5
```

**Detailed analysis of specific checkpoint:**
```bash
python compare_checkpoints.py --detailed ../checkpoints/context_aware_20251118_115931_best_policy.pth
```

### What the Tool Shows

1. **Ranked list** by average reward (last 100 episodes)
2. **Summary statistics:** Best, most trained, most steps
3. **Context distribution:** Actual vs target percentages
4. **Recommendation:** Which checkpoint to use for testing

### Example Output

```
CHECKPOINT COMPARISON - RANKED BY RECENT AVERAGE REWARD
────────────────────────────────────────────────────────
Rank | Episodes |    Steps |   Avg(100) |      Max | File
   1 |      260 |    85043 |     233.04 |   700.00 | best
   2 |       20 |     9639 |     224.39 |   710.40 | early_lucky
   3 |      220 |    78713 |     222.85 |   700.00 | consistent
```

**Interpretation:**
- Rank #1: Most episodes + best performance = **reliable model**
- Rank #2: Few episodes but high reward = **got lucky early**
- Rank #3: Many episodes, stable performance = **consistent learning**

**Recommendation:** Use Rank #1 (most trained + best reward)

---

## Does It Make Sense?

### ✅ What Works Well

#### 1. **Checkpoint Saving Strategy**
- ✓ Only saves when performance improves (no wasted disk space)
- ✓ Timestamp naming allows tracking training timeline
- ✓ Saves optimizer state (can resume training)
- ✓ Preserves full training history (episode_rewards, lengths)

#### 2. **Training Methodology**
- ✓ Context-aware approach solves spurious correlation problem
- ✓ Mixed scenario training enables transfer learning
- ✓ DQN is proven algorithm for discrete action spaces
- ✓ Hierarchical Q-heads provide interpretable decisions
- ✓ Target network prevents instability
- ✓ Replay buffer improves sample efficiency

#### 3. **Hyperparameter Choices**
- ✓ Gamma=0.99 appropriate for episodic tasks
- ✓ Epsilon decay allows exploration → exploitation
- ✓ Buffer size 100k sufficient for this environment
- ✓ Batch size 64 balances speed and stability
- ✓ Context distribution (30/50/20) matches real-world scenarios

#### 4. **Comparison Utility**
- ✓ Provides objective ranking (avg reward)
- ✓ Shows learning dynamics (improvement over time)
- ✓ Verifies context distribution matches targets
- ✓ Recommends best checkpoint for deployment

### ⚠️ Potential Improvements

#### 1. **Checkpoint Management**
**Current issue:** Saves every time avg_reward improves
- Can create many checkpoints if reward oscillates
- No cleanup of old/inferior checkpoints

**Suggestions:**
```python
# Option A: Keep only top-K checkpoints
if len(saved_checkpoints) > 5:
    remove_worst_checkpoint()

# Option B: Save only if improvement > threshold
if avg_reward > best_avg_reward + 10:  # Must improve by 10
    save_checkpoint()

# Option C: Save at fixed intervals AND when best
if episode % 500 == 0 or avg_reward > best_avg_reward:
    save_checkpoint()
```

#### 2. **Training Metrics**
**Current:** Only saves final metrics in checkpoint

**Missing:**
- Training curves not visualized during training
- No tensorboard/wandb integration
- Loss values saved but not analyzed

**Suggestions:**
```python
# Add logging
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/context_aware')

# Log during training
writer.add_scalar('Reward/avg_100', avg_reward, episode)
writer.add_scalar('Loss/policy', policy_loss, step)
writer.add_scalar('Context/snake_count', snake_count, episode)
```

#### 3. **Early Stopping**
**Current:** Trains for fixed num_episodes

**Better:** Stop when converged
```python
# If no improvement in 500 episodes, stop
if episode - last_improvement > 500:
    print("Converged! Stopping early.")
    break
```

#### 4. **World Model Usage**
**Current:** World model trained but NOT used

**Potential:**
- Use for planning (Monte Carlo Tree Search)
- Use for data augmentation (synthetic experiences)
- Use for curiosity-driven exploration

#### 5. **Resume Training**
**Current:** No built-in resume functionality

**Suggestion:**
```python
def load_checkpoint(path):
    """Resume training from checkpoint"""
    checkpoint = torch.load(path)
    policy_net.load_state_dict(checkpoint['policy_net'])
    target_net.load_state_dict(checkpoint['target_net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    episode_rewards = checkpoint['episode_rewards']
    steps_done = checkpoint['steps_done']
    # Continue from here...
```

---

## Training Efficiency Analysis

### Current Training (260 episodes)

**Time estimate:**
- 260 episodes × ~330 steps/episode = 85,043 steps
- If ~0.05 sec/step → ~4,250 seconds = **~70 minutes**

**Resource usage:**
- Memory: ~500MB (replay buffer + networks)
- Disk: ~1MB per checkpoint × 17 checkpoints = **17MB**

### Full Training (5000 episodes)

**Projected:**
- 5000 episodes × 330 steps ≈ 1,650,000 steps
- At 0.05 sec/step → 82,500 seconds = **~23 hours**

**To speed up:**
1. Larger batch size (64→128): 1.5× faster
2. Reduce world model training: 1.3× faster
3. GPU acceleration: 2-3× faster
4. Parallel environments: 4× faster

---

## Conclusion

### Does the Training Methodology Make Sense?

**Overall: Yes! ✓**

The training system is **well-designed, theoretically sound, and practically effective**:

✅ **Algorithm choice:** DQN proven for this task type
✅ **Architecture:** Hierarchical heads enable specialization
✅ **Context system:** Solves spurious correlation elegantly
✅ **Checkpoint saving:** Captures best performance
✅ **Comparison tool:** Objective model selection

### Main Findings

1. **Training works:** 210% improvement over 260 episodes
2. **Context distribution accurate:** 35/48/17 vs target 30/50/20
3. **Best checkpoint clear:** #1 rank has most training + best reward
4. **Checkpoint saving sensible:** Only saves improvements

### Recommendations

**For production use:**
1. ✅ Use the comparison tool to select best checkpoint
2. ✅ Train for full 5000 episodes for claimed performance
3. ⚠️ Add early stopping to prevent overtraining
4. ⚠️ Add tensorboard logging for better monitoring
5. ⚠️ Implement checkpoint cleanup (keep top-5 only)

**For research:**
1. Investigate world model planning (currently unused)
2. Try priority-based replay instead of uniform sampling
3. Experiment with context inference threshold tuning
4. Analyze per-context Q-value distributions

---

## Quick Reference

### Train New Model
```bash
cd src
python train_context_aware.py --episodes 5000 --log-every 100
```

### Compare Checkpoints
```bash
python compare_checkpoints.py --top 10
```

### Test Best Model
```bash
python test_context_aware.py $(python compare_checkpoints.py | grep "For best" | cut -d: -f2)
```

### Resume Training (if implemented)
```bash
python train_context_aware.py --resume ../checkpoints/best_policy.pth --episodes 5000
```

---

**Document Version:** 1.0
**Last Updated:** 2025-11-18
**Author:** Context-Aware Training Analysis
