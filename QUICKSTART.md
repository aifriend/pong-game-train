# 🚀 Quick Start Guide - PPO Pong Training

Get started training a Pong AI agent in **under 5 minutes**.

---

## ⚡ 3-Step Setup

### 1. Install

```bash
# Clone and enter directory
cd /path/to/Pong

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### 2. Train

```bash
# Run the 5-phase curriculum (~15 min on M1 Mac)
uv run python scripts/train_ppo_curriculum.py
```

### 3. Evaluate

```bash
# Test your trained agent
uv run python scripts/evaluate_agent.py --weights models/ppo_final.zip
```

---

## 📊 What to Expect

### Training Progress

```
Phase 1: Easy Wins (30 minutes) - Ball 0.7x
  → Agent learns that scoring = good
  → Win rate vs SLOW_AI: 0% → 80%+

Phase 2: Competitive (90 minutes) - Ball 0.9x
  → Agent learns offensive shot placement
  → Win rate vs NORMAL_AI: 0% → 50%+

Phase 3: Master (2 hours) - Ball 1.0x
  → Agent masters reactive opponent
  → Win rate vs REACTIVE_AI: 0% → 45%+
```

### Sample Output

```bash
$ uv run python scripts/train_ppo_curriculum.py

🏓 PPO TRAINING WITH 3-PHASE CURRICULUM
============================================================

Phases:
  → Phase 1: Easy Wins (50,000 steps)
    Phase 2: Competitive (150,000 steps)
    Phase 3: Master (200,000 steps)

============================================================
🎯 PHASE 1: Easy Wins
============================================================
   Opponent: slow_ai
   Ball Speed: 0.7x
   Target Win Rate: 80%
   Timesteps: 50,000
   Description: Learn that scoring = good against slow opponent

📦 Creating new PPO model...
Using cpu device

🚀 Starting Phase 1 training...
...
```

---

## 🎯 Quick Commands

```bash
# === Play Game ===
uv run pong          # Short form to play
uv run main.py       # Or use main.py

# === Training ===

# Full curriculum (recommended)
uv run python scripts/train_ppo_curriculum.py

# Train single phase
uv run python scripts/train_ppo_curriculum.py --phase 1 --single-phase

# Custom parallel environments
uv run python scripts/train_ppo_curriculum.py --envs 16

# === Evaluation ===

# Full benchmark
uv run python scripts/evaluate_agent.py --weights models/ppo_final.zip

# Specific opponent
uv run python scripts/evaluate_agent.py --weights models/ppo_final.zip --opponent normal_ai

# === Monitoring ===

# TensorBoard
uv run tensorboard --logdir ./tensorboard/

# === Play against AI ===

# Play against the trained PPO model
./play.sh
```

---

## 🔧 Customization

### Adjust Training Speed

```bash
# Faster (fewer parallel envs, less memory)
uv run python scripts/train_ppo_curriculum.py --envs 2

# Slower but more stable
uv run python scripts/train_ppo_curriculum.py --envs 16
```

### Train from Checkpoint

```bash
# Continue from Phase 2
uv run python scripts/train_ppo_curriculum.py --phase 2

# The script automatically loads the previous phase's model
```

### Custom Save Location

```bash
uv run python scripts/train_ppo_curriculum.py --save-dir ./my_models/ --tensorboard ./my_logs/
```

---

## 📈 Expected Results

### Training Time (M1 Mac, 8 parallel envs)

| Phase | Ball Speed | Steps | Time | Win Rate Target |
|-------|------------|-------|------|-----------------|
| 1 | 0.7x | 50K | 30 min | 80% vs SLOW_AI |
| 2 | 0.9x | 150K | 90 min | 50% vs NORMAL_AI |
| 3 | 1.0x | 200K | 2 hours | 45% vs REACTIVE_AI |
| **Total** | - | **400K** | **~4 hours** | **Master level** |

### Benchmarks

After full training, expect:

- ✅ **90%+ win rate** vs SLOW_AI (easy opponent)
- ✅ **55%+ win rate** vs NORMAL_AI (competitive)
- ✅ **45%+ win rate** vs REACTIVE_AI (master level)

---

## 🎮 Manual Play

Test the environment yourself:

```bash
# Play against the trained PPO model
./play.sh

# Or play the basic version
uv run python scripts/play.py

# Controls:
# - UP/DOWN ARROW: Move paddle
# - ESC: Quit
```

---

## 📚 Next Steps

- **[Full Documentation](README.md)** - Complete feature list
- **[Training Guide](docs/PPO_TRAINING.md)** - Deep dive into PPO approach
- **[SB3 Tutorial](learn/stable_baselines3_tutorial.md)** - Learn Stable-Baselines3

---

## 🐛 Common Issues

### `ModuleNotFoundError: No module named 'pong'`

**Solution**: Use `uv run` to run commands (it sets up the path automatically):

```bash
uv run python scripts/train_ppo_curriculum.py
```

### Training is very slow

**Solution**: Reduce parallel environments:

```bash
uv run python scripts/train_ppo_curriculum.py --envs 2
```

### Agent not improving

**Solution**: Check TensorBoard for learning curves:

```bash
uv run tensorboard --logdir ./tensorboard/
```

Look for:
- `rollout/ep_rew_mean` should be increasing
- `train/loss` should be decreasing
- `curriculum/win_rate` should approach target

---

<div align="center">

**Ready to train your Pong master? 🏓**

Run `uv run python scripts/train_ppo_curriculum.py` and let's go!

</div>
