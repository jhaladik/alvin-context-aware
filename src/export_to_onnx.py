"""
Export Context-Aware Agent to ONNX for Web Deployment

This script converts the PyTorch model to ONNX format for running in browsers
via ONNX.js. This enables zero-latency AI inference in web demos!

Usage:
    python export_to_onnx.py checkpoints/context_aware_advanced_20251118_173024_best_policy.pth
    python export_to_onnx.py <checkpoint> --output web/models/warehouse_ai.onnx
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))

import torch
import argparse
from context_aware_agent import ContextAwareDQN


class ONNXWrapper(torch.nn.Module):
    """
    Wrapper to export the agent in a format suitable for ONNX.

    The exported model will take a raw observation (95-dim vector) and output
    Q-values for each action. Context inference is built-in.
    """

    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    def forward(self, observation):
        """
        Forward pass for ONNX export.

        Args:
            observation: (batch, 95) - observation with context already added

        Returns:
            q_values: (batch, 4) - Q-values for UP, DOWN, LEFT, RIGHT
        """
        # Get combined Q-values from all heads
        q_values = self.agent.get_combined_q(observation)
        return q_values


def export_to_onnx(checkpoint_path, output_path="warehouse_ai_agent.onnx", opset_version=11):
    """
    Export trained agent to ONNX format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint (.pth)
        output_path: Where to save ONNX model
        opset_version: ONNX opset version (11 for broad compatibility)
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Create agent
    agent = ContextAwareDQN(obs_dim=95, action_dim=4)
    agent.load_state_dict(checkpoint['policy_net'])
    agent.eval()

    print(f"  Episodes trained: {len(checkpoint.get('episode_rewards', []))}")
    print(f"  Planning actions: {checkpoint.get('planning_count', 0)}")

    # Wrap for ONNX export
    wrapped_model = ONNXWrapper(agent)
    wrapped_model.eval()

    # Create dummy input (batch_size=1, obs_dim=95)
    dummy_input = torch.randn(1, 95)

    print(f"\nExporting to ONNX...")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: (1, 4) - Q-values for 4 actions")

    # Export to ONNX
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,  # Optimize
        input_names=['observation'],
        output_names=['q_values'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'q_values': {0: 'batch_size'}
        },
        verbose=False
    )

    # Check file size
    file_size_kb = os.path.getsize(output_path) / 1024
    file_size_mb = file_size_kb / 1024

    print(f"\n[SUCCESS] Export successful!")
    print(f"  Output: {output_path}")
    print(f"  Size: {file_size_kb:.1f} KB ({file_size_mb:.2f} MB)")
    print(f"  Opset: {opset_version}")

    # Verify export
    print(f"\nVerifying ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("[OK] ONNX model is valid!")

        # Print model info
        print(f"\nModel Info:")
        print(f"  Inputs: {[input.name for input in onnx_model.graph.input]}")
        print(f"  Outputs: {[output.name for output in onnx_model.graph.output]}")
        print(f"  Nodes: {len(onnx_model.graph.node)}")

    except ImportError:
        print("[WARNING] Install 'onnx' package to verify: pip install onnx")
    except Exception as e:
        print(f"[WARNING] Verification warning: {e}")

    print(f"\n[INFO] Usage in JavaScript:")
    print(f"""
    // Load model
    const session = await ort.InferenceSession.create('{output_path}');

    // Prepare input (observation with context)
    const obs = new Float32Array(95);  // Your observation
    const tensor = new ort.Tensor('float32', obs, [1, 95]);

    // Run inference
    const results = await session.run({{ observation: tensor }});
    const qValues = results.q_values.data;

    // Select best action
    const action = qValues.indexOf(Math.max(...qValues));
    // 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    """)

    return output_path


def test_onnx_inference(onnx_path):
    """
    Test ONNX model inference to ensure it works.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("[WARNING] Install 'onnxruntime' to test: pip install onnxruntime")
        return

    print(f"\n[TEST] Testing ONNX inference...")

    # Create session
    session = ort.InferenceSession(onnx_path)

    # Create test input
    test_obs = torch.randn(1, 95).numpy()

    # Run inference
    import time
    start = time.time()
    results = session.run(None, {'observation': test_obs})
    elapsed_ms = (time.time() - start) * 1000

    q_values = results[0]
    best_action = q_values.argmax()

    print(f"[OK] Inference successful!")
    print(f"  Input shape: {test_obs.shape}")
    print(f"  Output shape: {q_values.shape}")
    print(f"  Q-values: {q_values[0]}")
    print(f"  Best action: {best_action} ({'UP,DOWN,LEFT,RIGHT'.split(',')[best_action]})")
    print(f"  Inference time: {elapsed_ms:.2f}ms")

    # Benchmark
    print(f"\n[BENCHMARK] Testing (100 inferences)...")
    times = []
    for _ in range(100):
        start = time.time()
        session.run(None, {'observation': test_obs})
        times.append((time.time() - start) * 1000)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"  Average: {avg_time:.2f}ms")
    print(f"  Min: {min_time:.2f}ms")
    print(f"  Max: {max_time:.2f}ms")
    print(f"  FPS potential: {1000/avg_time:.0f} FPS")


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch model to ONNX')
    parser.add_argument('checkpoint', help='Path to PyTorch checkpoint (.pth)')
    parser.add_argument('--output', default='warehouse_ai_agent.onnx',
                       help='Output ONNX file path')
    parser.add_argument('--opset', type=int, default=11,
                       help='ONNX opset version (11 for compatibility)')
    parser.add_argument('--test', action='store_true',
                       help='Test ONNX inference after export')

    args = parser.parse_args()

    # Export
    onnx_path = export_to_onnx(args.checkpoint, args.output, args.opset)

    # Test if requested
    if args.test:
        test_onnx_inference(onnx_path)

    print(f"\n[READY] Model ready for web deployment!")
    print(f"   Upload {args.output} to your web server")
    print(f"   Include ONNX.js runtime: https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js")


if __name__ == '__main__':
    main()
