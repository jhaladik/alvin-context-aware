# Setup Guide - Context-Aware Agent

## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Git (for cloning the repository)

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/[your-username]/context-aware-agent.git
cd context-aware-agent
```

### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
cd src
python -c "import torch; import numpy; import pygame; print('✓ All dependencies installed')"
```

### 5. Test Import Structure
```bash
python -c "import sys; sys.path.insert(0, 'core'); from context_aware_agent import ContextAwareDQN; print('✓ Imports work correctly')"
```

## Quick Test Run

### Option A: Use Pre-trained Model (Fastest)
```bash
cd src
# List available checkpoints
ls ../checkpoints/context_aware_*.pth

# Run visual test with a pre-trained model
python context_aware_visual_games.py ../checkpoints/context_aware_20251118_113247_best_policy.pth --game snake
```

### Option B: Train Your Own Model (2-3 hours)
```bash
cd src

# Quick training run (30 episodes, ~2 minutes)
python train_context_aware.py --episodes 30 --log-every 10

# Full training (5000 episodes, 2-3 hours)
python train_context_aware.py --episodes 5000 --log-every 100
```

### Option C: Command-Line Testing
```bash
cd src
python test_context_aware.py ../checkpoints/context_aware_20251118_113247_best_policy.pth
```

## Directory Structure Check

After installation, verify you have:
```
context-aware-agent/
├── src/
│   ├── context_aware_agent.py          ✓ Model architecture
│   ├── train_context_aware.py          ✓ Training script
│   ├── test_context_aware.py           ✓ Testing script
│   ├── context_aware_visual_games.py   ✓ Visual testing
│   └── core/
│       ├── temporal_observer.py        ✓ Observation system
│       ├── temporal_env.py             ✓ Environment
│       ├── world_model.py              ✓ World model
│       └── planning_test_games.py      ✓ Test games
├── docs/                               ✓ Documentation
├── checkpoints/                        ✓ Pre-trained models
├── README.md                           ✓ Main docs
└── requirements.txt                    ✓ Dependencies
```

## Troubleshooting

### Import Error: "No module named 'core'"
**Solution**: Run scripts from the `src/` directory
```bash
cd src
python train_context_aware.py  # ✓ Correct
```

### PyTorch Installation Issues
**Windows**:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Linux/Mac**:
```bash
pip install torch
```

### Pygame Display Issues
**Linux**: Install SDL dependencies
```bash
sudo apt-get install python3-pygame
```

**Mac**: Use Homebrew
```bash
brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf
```

### Permission Denied on Checkpoints
```bash
chmod +r checkpoints/*.pth
```

## Verification Tests

### Test 1: Dependencies
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import numpy; print(f'NumPy {numpy.__version__}')"
python -c "import pygame; print(f'Pygame {pygame.__version__}')"
```

### Test 2: Model Loading
```bash
cd src
python -c "
from context_aware_agent import ContextAwareDQN
import torch
agent = ContextAwareDQN(obs_dim=95, action_dim=4)
print(f'✓ Model created: {sum(p.numel() for p in agent.parameters()):,} parameters')
"
```

### Test 3: Environment
```bash
cd src
python -c "
import sys
sys.path.insert(0, 'core')
from temporal_env import TemporalRandom2DEnv
env = TemporalRandom2DEnv(grid_size=(20, 20), num_entities=3, num_rewards=10)
obs = env.reset()
print(f'✓ Environment works: observation shape {obs.shape}')
"
```

### Test 4: Visual Games
```bash
cd src
# Should open pygame window (close it after verifying)
python context_aware_visual_games.py ../checkpoints/context_aware_20251118_113247_best_policy.pth --game snake
```

## Next Steps

Once setup is complete:

1. **Read the quick start**: `docs/CONTEXT_AWARE_QUICKSTART.md`
2. **Understand the architecture**: `docs/CONTEXT_AWARE_TRAINING.md`
3. **Run visual test**: See the agent in action
4. **Train your model**: Start with 30 episodes, then scale up
5. **Experiment**: Try different hyperparameters and environments

## Performance Expectations

### Hardware Requirements
- **Minimum**: CPU with 4GB RAM
- **Recommended**: CPU with 8GB RAM or GPU (optional)
- **Training time**: 2-3 hours for 5000 episodes (CPU)

### Expected Results After Training
- **Snake**: 6.0+ average score (was 0.0)
- **Pac-Man**: 3.0+ average score
- **Dungeon**: 2.0+ average score
- **Context detection**: >80% accuracy

## Getting Help

If you encounter issues:

1. Check this setup guide
2. Read `docs/CONTEXT_AWARE_QUICKSTART.md`
3. Check the troubleshooting section in `README.md`
4. Open a GitHub issue with:
   - Error message
   - Python version (`python --version`)
   - OS and version
   - Steps to reproduce

---

**Ready to go?**
```bash
cd src
python context_aware_visual_games.py ../checkpoints/context_aware_20251118_113247_best_policy.pth
```
