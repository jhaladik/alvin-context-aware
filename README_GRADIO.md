# 🤖 Context-Aware Foundation Agent - Interactive Demo

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/YOUR_USERNAME/context-aware-agent)

**Interactive demonstration** of a context-aware reinforcement learning agent with faith-based exploration, model-based planning, and automatic strategy adaptation.

## 🎯 Key Features

### Context Adaptation
- Automatically switches between **collection**, **balanced**, and **survival** strategies
- Infers context from temporal observation patterns
- Adapts behavior to environmental demands

### Faith-Based Exploration
- Discovers **hidden mechanics** through persistent exploration
- 80% mechanic discovery rate on unseen scenarios
- Maintains exploration despite negative feedback

### Model-Based Planning
- Plans **20 steps ahead** using learned world model
- +58% performance improvement over reactive baseline
- Enables strategic decision-making

### Zero-Shot Transfer
- **No domain-specific training** for warehouse scenarios
- Discovers 4/5 hidden mechanics without prior knowledge
- Strong generalization capability

## 📊 Performance Highlights

| Environment | Performance | Status |
|------------|-------------|---------|
| **Pac-Man** | 17.62 avg score | ⭐ Excellent |
| **Snake** | 2.98 avg score | ✅ Good |
| **Warehouse Discovery** | 80% mechanics | ⭐ Strong |

## 🎮 How to Use

1. **Choose Environment**: Select Pac-Man, Snake, or Warehouse scenario
2. **Configure Agent**:
   - **Faith Frequency**: 0-50% (exploration rate)
   - **Planning Frequency**: 0-50% (model-based planning)
   - **Planning Horizon**: 5-30 steps (lookahead depth)
3. **Run**:
   - **Step**: Execute one action
   - **Run Episode**: Complete full episode
   - **Reset**: Start new episode

## 🎨 Visual Guide

- 🟣 **Magenta Agent** = Faith action (exploration)
- 🔵 **Cyan Agent** = Planning action (model-based)
- 🟡 **Yellow Agent** = Reactive action (policy-based)

## 🔧 Recommended Configurations

### Best Performance
```
Faith: 0%
Planning: 20%
Horizon: 20 steps
```
*Use for maximum score and efficiency*

### Exploration Mode
```
Faith: 30%
Planning: 0%
Horizon: 20 steps
```
*Use for discovering hidden mechanics*

### Balanced
```
Faith: 15%
Planning: 15%
Horizon: 20 steps
```
*Use for semi-familiar environments*

## 🏗️ Architecture

### Fixed World Model
- **Input**: 180 observation dims + 3 context dims
- **Output**: 180 next observation (context passed through)
- **Benefit**: Eliminates bottleneck by not predicting constant context

### Expanded Temporal Observer
- **16 rays × 15 tiles** = 180 observation dims
- **Multi-scale windows**: Micro (5 frames), Meso (20 frames), Macro (50 frames)
- **Context inference**: Automatic from observation patterns

### Training Results
- **700 episodes** trained
- **684.55 avg reward** achieved
- **178 faith discoveries** during training

## 🎯 Warehouse Scenarios

### Hidden Shortcut
- **Mechanic**: Walls become passable when supervisor is far away
- **Discovery**: 100% (1/1)
- **Performance**: 10.65 avg packages

### Priority Zone
- **Mechanics**: Time-sensitive packages with decay and chain bonuses
- **Discovery**: 100% (3/3)
- **Performance**: 6.30 avg packages

### Charging Station
- **Mechanic**: Optimal battery management timing
- **Discovery**: 0% (needs more exploration)
- **Performance**: 4.25 avg packages

## 📚 Technical Details

### Model Architecture
- **Context-Aware DQN** with specialized Q-heads per context
- **Fixed World Model** (eliminates bottleneck)
- **Expanded Temporal Observer** (16×15 rays)

### Key Innovations
1. **Context Separation**: World model doesn't predict constant context
2. **Faith System**: Persistent exploration despite negative feedback
3. **Multi-Scale Temporal**: Captures patterns across different time scales

## 🚀 Deployment

This demo runs on **HuggingFace Spaces** with:
- Python backend (PyTorch inference)
- Gradio frontend (auto-generated UI)
- Real-time interaction (no JavaScript lag)

## 📖 Citation

```bibtex
@software{context_aware_agent_2025,
  title={Context-Aware Foundation Agent with Faith-Based Exploration},
  author={Your Name},
  year={2025},
  url={https://github.com/YOUR_USERNAME/alvin-context-aware}
}
```

## 🔗 Links

- [GitHub Repository](https://github.com/YOUR_USERNAME/alvin-context-aware)
- [Technical Documentation](https://github.com/YOUR_USERNAME/alvin-context-aware/blob/main/COMPREHENSIVE_TEST_RESULTS.md)
- [Architecture Details](https://github.com/YOUR_USERNAME/alvin-context-aware/blob/main/WORLD_MODEL_BOTTLENECK_FIX.md)

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

**Built with** ❤️ **using PyTorch, Gradio, and HuggingFace Spaces**
