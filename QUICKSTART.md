# 🚀 Quick Start Guide

Get started with Pong RL Training Environment in 5 minutes!

## ⚡ Installation (2 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/your-username/pong-rl-env.git
cd pong-rl-env

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import pong, gymnasium, torch; print('✓ Ready to train!')"
```

## 🎯 Train Your First Agent (3 minutes)

```bash
# Start training (runs for 1000 episodes)
PYTHONPATH=. python trainer/main.py
```

**What's happening:**
- Agent learns to play Pong through trial and error
- Checkpoints saved every 100 episodes to `checkpoints/`
- Training stats displayed in terminal
- TensorBoard logs saved to `tensorboard_dqn/`

**Expected output:**
```
Episode 1/1000 | Score: -11.0 | Win: False | Epsilon: 1.00
Episode 100/1000 | Score: -5.3 | Win: False | Epsilon: 0.87
Episode 500/1000 | Score: 0.2 | Win: True | Epsilon: 0.50
Episode 1000/1000 | Score: 4.1 | Win: True | Epsilon: 0.10
```

## 🎮 Test Your Agent

```bash
# Play against your trained agent
PYTHONPATH=. python scripts/play_against_agent.py checkpoints/checkpoint_episode_1000.pth
```

**Controls:**
- **W**: Move up
- **S**: Move down
- **ESC**: Quit

## 📊 Monitor Training (Optional)

```bash
# In a new terminal, start TensorBoard
tensorboard --logdir ./tensorboard_dqn/

# Open http://localhost:6006 in your browser
```

## 🎓 Next Steps

### Experiment with Different Algorithms

```bash
# Try Stable-Baselines3 PPO
PYTHONPATH=. python scripts/train_pong_agent.py
```

### Customize Training

Edit `trainer/config.py` to change:
- Learning rate
- Episode count
- Network architecture
- Reward function

### Explore Documentation

- `learn/README.md` - Comprehensive tutorials
- `docs/README_GYM.md` - Environment specifications
- `CONTRIBUTING.md` - How to contribute

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'pong'` | Use `PYTHONPATH=. python ...` |
| Training is slow | Use headless mode (default) |
| pygame errors | `pip install pygame==2.1.2` |
| Display not found | You're good! Headless mode doesn't need display |

## 💡 Pro Tips

1. **Start small**: Train for 100 episodes first to verify setup
2. **Monitor progress**: Use TensorBoard to visualize learning
3. **Test checkpoints**: Compare different training stages
4. **GPU acceleration**: Install CUDA PyTorch for 10x speedup
5. **Headless mode**: Always use for serious training (it's the default)

## 📞 Need Help?

- 📖 Read the [full README](README.md)
- 🐛 Check [existing issues](https://github.com/your-username/pong-rl-env/issues)
- 💬 Start a [discussion](https://github.com/your-username/pong-rl-env/discussions)

---

**Happy training! 🎮🤖**

