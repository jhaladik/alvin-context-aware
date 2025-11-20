# 🚀 HuggingFace Spaces Deployment Guide

Complete guide to deploying the Context-Aware Foundation Agent demo to HuggingFace Spaces.

---

## 📋 Prerequisites

1. **HuggingFace Account**: Sign up at https://huggingface.co/join
2. **Git**: Installed on your system
3. **Model Files**: Trained model checkpoint (included in repo)

---

## 🎯 Quick Deployment (5 Minutes)

### Step 1: Create New Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Fill in:
   - **Name**: `context-aware-agent` (or your preferred name)
   - **License**: MIT
   - **SDK**: Gradio
   - **Hardware**: CPU Basic (free tier) or GPU if available
4. Click "Create Space"

### Step 2: Clone Your Space

```bash
# Clone the HuggingFace Space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/context-aware-agent
cd context-aware-agent
```

### Step 3: Copy Files

Copy these files from your project to the Space directory:

```bash
# Essential files
cp /path/to/alvin-context-aware/app.py .
cp /path/to/alvin-context-aware/gradio_demo.py .
cp /path/to/alvin-context-aware/requirements.txt .
cp /path/to/alvin-context-aware/README_GRADIO.md ./README.md

# Copy source code
cp -r /path/to/alvin-context-aware/src .

# Copy model checkpoint
mkdir -p checkpoints
cp /path/to/alvin-context-aware/checkpoints/faith_fixed_20251120_162417_final_policy.pth checkpoints/
cp /path/to/alvin-context-aware/checkpoints/faith_fixed_20251120_162417_world_model.pth checkpoints/
```

### Step 4: Push to HuggingFace

```bash
git add .
git commit -m "Initial deployment of Context-Aware Foundation Agent"
git push
```

### Step 5: Done!

Your Space will automatically build and deploy. Visit:
```
https://huggingface.co/spaces/YOUR_USERNAME/context-aware-agent
```

---

## 📁 Required Files Structure

```
context-aware-agent/
├── app.py                    # Entry point
├── gradio_demo.py           # Main demo code
├── requirements.txt         # Python dependencies
├── README.md                # Space description
├── src/                     # Source code
│   ├── context_aware_agent.py
│   ├── warehouse_faith_scenarios.py
│   └── core/
│       ├── expanded_temporal_observer.py
│       ├── planning_test_games.py
│       ├── context_aware_world_model.py
│       └── ...
└── checkpoints/            # Model files
    ├── faith_fixed_20251120_162417_final_policy.pth
    └── faith_fixed_20251120_162417_world_model.pth
```

---

## ⚙️ Configuration Options

### Hardware Selection

**CPU Basic (Free)**
- Sufficient for demo
- ~2-3 second inference time
- Good for showcasing

**GPU (Paid)**
- Faster inference (<1 second)
- Better user experience
- Recommended for high traffic

### Environment Variables (Optional)

Add in Space settings if needed:
```
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

---

## 🎨 Customization

### Update README

Edit `README.md` in your Space to:
- Add your HuggingFace username
- Include your GitHub repo link
- Add screenshots/GIFs
- Update citation with your name

### Modify Demo

Edit `gradio_demo.py` to:
- Change default game/scenario
- Adjust UI theme (`gr.themes.Soft()` → `gr.themes.Base()`)
- Add more information panels
- Include additional metrics

### Update Requirements

Edit `requirements.txt` if needed:
- Specific PyTorch version for GPU
- Additional dependencies
- Version constraints

---

## 🐛 Troubleshooting

### Build Fails

**Issue**: "Module not found"
```bash
# Check requirements.txt includes all dependencies
torch>=1.9.0
numpy>=1.19.0
gradio>=4.12.0
pillow>=10.0.0
```

**Issue**: "File not found: checkpoints/..."
```bash
# Ensure model files are committed
git lfs install
git lfs track "*.pth"
git add checkpoints/*.pth
git commit -m "Add model checkpoints"
git push
```

### Slow Performance

**Solution 1**: Reduce inference complexity
- Lower planning horizon (20 → 10)
- Reduce episode max steps (200 → 100)

**Solution 2**: Upgrade to GPU hardware
- Go to Space Settings → Hardware
- Select T4 Small or better

### Memory Issues

**Solution**: Optimize model loading
```python
# In gradio_demo.py, add:
torch.set_num_threads(2)  # Limit CPU threads
```

---

## 🔍 Testing Locally

Before deploying, test locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo
python app.py

# Open browser to http://127.0.0.1:7860
```

**Check**:
- ✅ UI loads correctly
- ✅ Game renders properly
- ✅ Actions execute without errors
- ✅ Info panel updates
- ✅ Episode resets work

---

## 📊 Monitoring

### View Logs

1. Go to your Space page
2. Click "Logs" tab
3. Monitor for errors

### Analytics

HuggingFace provides:
- View count
- Unique visitors
- Like count
- Fork count

---

## 🚀 Advanced Deployment

### Custom Domain

1. Go to Space Settings
2. Add custom domain (Pro account required)
3. Configure DNS

### Private Space

1. Space Settings → Visibility
2. Select "Private"
3. Manage access tokens

### Persistent Storage

For user-specific data:
```python
# In gradio_demo.py
demo.launch(
    enable_queue=True,
    persist_folder="./user_data"
)
```

---

## 📝 Deployment Checklist

Before going live:

- [ ] Model files uploaded and working
- [ ] README.md updated with your info
- [ ] All dependencies in requirements.txt
- [ ] Tested locally successfully
- [ ] Committed all necessary files
- [ ] Removed any sensitive data
- [ ] Added appropriate license
- [ ] Updated links in README
- [ ] Tested on HF Spaces (after push)
- [ ] Shared on social media / communities

---

## 🌟 Promotion Tips

### Share Your Space

1. **Twitter/X**: Share with #HuggingFace #MachineLearning #ReinforcementLearning
2. **LinkedIn**: Post about your project
3. **Reddit**: r/MachineLearning, r/reinforcementlearning
4. **HuggingFace Hub**: Add tags: `reinforcement-learning`, `pytorch`, `planning`

### Add to Model Card

Include:
- Performance metrics
- Architecture diagram
- Training details
- Use cases
- Citation information

### Create Video Demo

Record:
- Pac-Man gameplay
- Warehouse discovery
- Configuration changes
- Faith/planning actions

---

## 🔗 Useful Links

- **HuggingFace Spaces Docs**: https://huggingface.co/docs/hub/spaces
- **Gradio Documentation**: https://gradio.app/docs/
- **Git LFS**: https://git-lfs.github.com/
- **Your Space**: https://huggingface.co/spaces/YOUR_USERNAME/context-aware-agent

---

## 💡 Tips for Success

1. **Start with CPU**: Free tier is fine for demonstration
2. **Clear Documentation**: Well-written README attracts more users
3. **Visual Appeal**: Add screenshots and GIFs to README
4. **Responsive**: Respond to comments and questions
5. **Iterate**: Update based on user feedback

---

## 🎯 Next Steps

After successful deployment:

1. **Monitor Usage**: Check logs and analytics
2. **Gather Feedback**: Engage with users in comments
3. **Iterate**: Improve based on feedback
4. **Promote**: Share on social media and communities
5. **Extend**: Add new features or scenarios

---

## 🏆 Success Metrics

Track these to measure impact:

- **Views**: How many people visited
- **Likes**: User appreciation
- **Forks**: Others building on your work
- **Comments**: Community engagement
- **Shares**: Viral potential

**Goal**: 100+ views in first week, 10+ likes in first month

---

## 📞 Support

If you encounter issues:

1. Check [HuggingFace Forums](https://discuss.huggingface.co/)
2. Review [Gradio Discord](https://discord.gg/gradio)
3. Open GitHub issue in your repo
4. Contact HuggingFace support

---

**Happy Deploying!** 🚀

*Last updated: November 20, 2025*
