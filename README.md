<div align="center">

# 🏓 Pong AI - PPO Training Environment

### Train AI agents to master Pong using PPO and win-focused reinforcement learning

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.0%2B-green.svg)](https://stable-baselines3.readthedocs.io/)

[Features](#-features) • [Quick Start](#-quick-start) • [Training](#-training) • [Evaluation](#-evaluation)

</div>

---

## 📖 About

A production-ready reinforcement learning environment for training AI agents to **win** at Pong using **PPO (Proximal Policy Optimization)**. Features a clean, headless environment optimized for fast training with win-focused rewards.

**Key Insight**: Previous approaches failed because they rewarded rallying instead of winning. This implementation focuses purely on scoring and winning games.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🚀 **Headless Training** | Pure Python, no pygame required - 10x faster |
| 🎯 **Win-Focused Rewards** | Agent learns to score, not just rally |
| 🧠 **PPO Algorithm** | Stable-Baselines3 PPO optimized for sparse rewards |
| 📊 **5-Phase Curriculum** | Progressive difficulty: Easy → Beginner → Medium → Competitive → Master |
| ⚡ **Parallel Training** | 8 environments running simultaneously |
| 💾 **Auto Checkpoints** | Models saved throughout training |
| 📈 **TensorBoard Logging** | Real-time training metrics visualization |

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd Pong

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Train Agent

```bash
# Activate environment
source .venv/bin/activate

# Run full 5-phase curriculum
PYTHONPATH=. python scripts/train_ppo_curriculum.py

# Or train single phase
PYTHONPATH=. python scripts/train_ppo_curriculum.py --phase 1 --single-phase
```

### 🔄 Resuming Training

If you've already started training and want to continue from where you left off:

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Check which phases are completed
ls models/ | grep "ppo_phase.*_final.zip"

# 3. Resume from a specific phase (automatically loads previous phase model)
PYTHONPATH=. python scripts/train_ppo_curriculum.py --phase 3

# Or continue full curriculum (starts from phase 1, but loads existing models)
PYTHONPATH=. python scripts/train_ppo_curriculum.py
```

**Note**: The training script automatically loads the previous phase's model when you specify `--phase` with a phase number > 1. For example, `--phase 3` will automatically load `ppo_phase2_final.zip` if it exists.

### Evaluate Trained Agent

```bash
# Full evaluation against all opponents
PYTHONPATH=. python scripts/evaluate_agent.py --weights models/ppo_final.zip

# Test against specific opponent
PYTHONPATH=. python scripts/evaluate_agent.py --weights models/ppo_final.zip --opponent reactive_ai
```

### Monitor Training

```bash
# Start TensorBoard (in separate terminal)
tensorboard --logdir ./tensorboard/

# Open http://localhost:6006
```

---

## 🎯 Training

### 5-Phase Curriculum

The agent progresses through 5 phases with ~10% opponent speed increments, graduated hit rewards, and decaying hyperparameters:

| Phase | Opponent | Ball Speed | Steps | Target WR | LR | Hit Reward |
|-------|----------|------------|-------|-----------|----|------------|
| **1: Easy Wins** | SLOW_AI (40%) | 0.6x | 100K | 60% | 3e-4 | 0.5 |
| **2: Beginner** | BEGINNER_AI (55%) | 0.7x | 100K | 50% | 2.5e-4 | 0.3 |
| **3: Medium** | MEDIUM_AI (65%) | 0.8x | 150K | 40% | 1.5e-4 | 0.1 |
| **4: Competitive** | NORMAL_AI (70%) | 0.9x | 200K | 30% | 5e-5 | 0.05 |
| **5: Master** | REACTIVE_AI (70%+pred) | 1.0x | 200K | 20% | 3e-5 | 0.05 |

**Total: 750,000 base steps** (~15 min on M1 Mac). Phase 5 uses mixed-opponent training (4 REACTIVE + 2 NORMAL + 2 MEDIUM environments).

**Anti-regression**: Training automatically stops if the agent regresses on easy opponents (catastrophic forgetting detection).

### Graduated Reward Shaping

| Event | Reward | Purpose |
|-------|--------|---------|
| Score a point | +5 | Primary objective |
| Opponent scores | -5 | Penalty |
| Win game | +10 | Ultimate goal |
| Lose game | -10 | Strong negative signal |
| Hit ball | +0.5 → +0.05 | Graduated: bootstraps motor skills, then fades |
| Pressure opponent | +0.15 → +0.08 | Offensive shot placement shaping |
| Per step | -0.001 | Faster games |

**Key**: Hit rewards fade gradually across phases. Strong pressure shaping (0.15) guides offensive play.

---

## 📊 Evaluation

### Best Achieved Results

| Opponent | Win Rate | Description |
|----------|----------|-------------|
| SLOW_AI | 93% | Dominates |
| BEGINNER_AI | 90% | Strong |
| MEDIUM_AI | 77% | Good offensive play |
| NORMAL_AI | 43% | Competitive |
| REACTIVE_AI | 3% | Still challenging |

### Example Evaluation

```bash
$ PYTHONPATH=. python scripts/evaluate_agent.py --weights models/ppo_final.zip --episodes 30
```

---

## 🎮 Environment Details

### Observation Space (9 dimensions, normalized)

```
[ball_x, ball_y, ball_vx, ball_vy, player_y, opponent_y, ball_dist, player_score, opp_score]
```

All values normalized to [-1, 1] or [0, 1] for stable neural network training.

### Action Space

- **0**: Stay (no movement)
- **1**: Move paddle up
- **2**: Move paddle down

### Opponent Types

| Type | Speed | Dead Zone | Behavior |
|------|-------|-----------|----------|
| SLOW_AI | 40% | 35px | Forgiving, good for learning |
| BEGINNER_AI | 55% | 28px | Moderate challenge |
| MEDIUM_AI | 65% | 22px | Competent opponent |
| NORMAL_AI | 70% | 20px | Fast and competitive |
| REACTIVE_AI | 70% | 18px | Late ball trajectory prediction (last 30% of court) |

---

## 📁 Project Structure

```
Pong/
├── pong/
│   ├── env/
│   │   ├── pong_headless.py      # Headless environment (PPO-optimized)
│   │   ├── pong_gym_env.py       # Pygame environment (manual play)
│   │   └── wrappers.py           # Reward wrappers
│   ├── game/                     # Game physics
│   │   ├── ball.py
│   │   ├── player.py
│   │   ├── opponent.py
│   │   └── game_manager.py
│   └── constants.py
├── scripts/
│   ├── train_ppo_curriculum.py   # Main training script
│   ├── evaluate_agent.py         # Evaluation suite
│   └── play.py                   # Manual play
├── tests/                        # Unit tests
├── docs/
│   ├── PPO_TRAINING.md          # Detailed training guide
│   └── PPO_TRAINING.md          # Detailed training guide
├── learn/
│   ├── environment_guide.md
│   └── stable_baselines3_tutorial.md
└── resources/                    # Game assets
```

---

## 🔧 Advanced Usage

### Custom Training Configuration

```python
from scripts.train_ppo_curriculum import train_phase
from stable_baselines3 import PPO

# Train with custom parameters
model = train_phase(
    phase=1,
    n_envs=16,           # More parallel environments
    save_dir="./my_models/",
    verbose=1
)
```

### Create Custom Environment

```python
from pong.env.wrappers import make_ppo_env
from pong.env.pong_headless import OpponentType

# Create environment with custom settings
env = make_ppo_env(
    opponent_type=OpponentType.NORMAL_AI,
    ball_speed=1.2,      # Faster ball
    max_score=11,        # Longer games
    point_reward=15.0,   # Higher rewards
)
```

### Load and Use Trained Model

```python
from stable_baselines3 import PPO
import gymnasium as gym

# Load model
model = PPO.load("models/ppo_final.zip")

# Play episode
env = gym.make("PongHeadless-v0")
obs, _ = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break
```

---

## 📚 Documentation

- **[PPO Training Guide](docs/PPO_TRAINING.md)** - Detailed training methodology
- **[Environment Guide](learn/environment_guide.md)** - Environment API reference
- **[SB3 Tutorial](learn/stable_baselines3_tutorial.md)** - Stable-Baselines3 usage

---

## 🧪 Testing

```bash
# Test environment
PYTHONPATH=. python pong/env/pong_headless.py

# Run test suite
PYTHONPATH=. python -m pytest tests/ -v

# Quick smoke test
PYTHONPATH=. python -c "
from pong.env.wrappers import make_ppo_env
from pong.env.pong_headless import OpponentType
env = make_ppo_env(OpponentType.SLOW_AI)
obs, _ = env.reset()
print(f'✓ Environment working! Obs shape: {obs.shape}')
"
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'pong'` | Use `PYTHONPATH=.` before commands |
| Training very slow | Increase `--envs` or check CPU usage |
| Out of memory | Reduce `--envs` to 2-4 |
| Agent not improving | Check TensorBoard for learning curves |
| Evaluation fails | Ensure model path includes/excludes `.zip` correctly |

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[Stable-Baselines3](https://stable-baselines3.readthedocs.io/)** - RL algorithm implementations
- **[Gymnasium](https://gymnasium.farama.org/)** - Environment interface standard
- **[PyTorch](https://pytorch.org/)** - Deep learning framework
- Schulman et al. for the PPO algorithm (2017)

---

<div align="center">

### ⭐ Star this repository if you find it useful!

**Happy Training! 🎮🤖**

</div>
