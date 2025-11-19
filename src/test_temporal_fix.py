"""
Quick test to verify temporal enhancement fixes work
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from context_aware_agent import ContextAwareDQN
from core.temporal_buffer_enhancement import TemporalBufferEnhancement

print("="*60)
print("TESTING TEMPORAL ENHANCEMENT FIXES")
print("="*60)

# Test 1: TemporalBufferEnhancement with single observation
print("\n1. Testing TemporalBufferEnhancement with single observation...")
enhancement = TemporalBufferEnhancement(obs_dim=95, buffer_size=50)

# Create a single observation
single_obs = torch.randn(95)
print(f"   Input shape: {single_obs.shape}")

# Add to buffer and enhance (before buffer is full)
enhancement.update_buffers(single_obs)
enhanced, uncertainty = enhancement.enhance_observation(single_obs)
print(f"   Enhanced shape: {enhanced.shape}")
print(f"   Uncertainty: {uncertainty.item():.4f}")
assert enhanced.shape == torch.Size([95]), f"Wrong shape: {enhanced.shape}"
print("   [OK] Single observation works!")

# Test 2: TemporalBufferEnhancement with batch
print("\n2. Testing TemporalBufferEnhancement with batch...")
batch_obs = torch.randn(32, 95)
print(f"   Input shape: {batch_obs.shape}")

enhanced_batch, uncertainty_batch = enhancement.enhance_observation(batch_obs)
print(f"   Enhanced shape: {enhanced_batch.shape}")
print(f"   Uncertainty: {uncertainty_batch.item():.4f}")
assert enhanced_batch.shape == torch.Size([32, 95]), f"Wrong shape: {enhanced_batch.shape}"
print("   [OK] Batch observation works!")

# Test 3: Fill buffer and test again
print("\n3. Testing with filled buffer...")
for i in range(60):
    enhancement.update_buffers(torch.randn(95))

single_obs2 = torch.randn(95)
enhanced2, uncertainty2 = enhancement.enhance_observation(single_obs2)
print(f"   Single enhanced shape: {enhanced2.shape}")
print(f"   Uncertainty (full buffer): {uncertainty2.item():.4f}")
assert enhanced2.shape == torch.Size([95]), f"Wrong shape: {enhanced2.shape}"

batch_obs2 = torch.randn(32, 95)
enhanced_batch2, uncertainty_batch2 = enhancement.enhance_observation(batch_obs2)
print(f"   Batch enhanced shape: {enhanced_batch2.shape}")
print(f"   Uncertainty (full buffer): {uncertainty_batch2.item():.4f}")
assert enhanced_batch2.shape == torch.Size([32, 95]), f"Wrong shape: {enhanced_batch2.shape}"
print("   [OK]Full buffer works!")

# Test 4: Integration with TemporalEnhancedAgent
print("\n4. Testing TemporalEnhancedAgent...")
from train_temporal_enhanced import TemporalEnhancedAgent

base_agent = ContextAwareDQN(obs_dim=95, action_dim=4)
enhanced_agent = TemporalEnhancedAgent(base_agent, freeze_base=True)

# Test get_q_values with single observation
single_obs3 = torch.randn(95)
q_vals, unc = enhanced_agent.get_q_values(single_obs3)
print(f"   Single obs Q-values shape: {q_vals.shape}")
assert q_vals.shape == torch.Size([1, 4]), f"Wrong Q shape: {q_vals.shape}"
print("   [OK]Single observation Q-values work!")

# Test get_q_values with batch
batch_obs3 = torch.randn(32, 95)
q_vals_batch, unc_batch = enhanced_agent.get_q_values(batch_obs3)
print(f"   Batch obs Q-values shape: {q_vals_batch.shape}")
assert q_vals_batch.shape == torch.Size([32, 4]), f"Wrong Q shape: {q_vals_batch.shape}"
print("   [OK]Batch observation Q-values work!")

# Test 5: Backward pass (gradient computation)
print("\n5. Testing gradient computation...")
loss = q_vals_batch.mean()
loss.backward()
print("   [OK]Backward pass works!")

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)
print("\nThe temporal enhancement is ready for training!")
print("You can now run the full 100-episode training.")
