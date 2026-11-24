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

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### Run the Game

```bash
# Short form (play Pong)
uv run pong

# Or: uv run main.py  |  uv run -m pong  |  uv run pong-play
```

### Train Agent

```bash
# Run full 5-phase curriculum
uv run python scripts/train_ppo_curriculum.py

# Or train single phase
uv run python scripts/train_ppo_curriculum.py --phase 1 --single-phase
```

### 🔄 Resuming Training

If you've already started training and want to continue from where you left off:

```bash
# 1. Check which phases are completed
ls models/ | grep "ppo_phase.*_final.zip"

# 2. Resume from a specific phase (automatically loads previous phase model)
uv run python scripts/train_ppo_curriculum.py --phase 3

# Or continue full curriculum (starts from phase 1, but loads existing models)
uv run python scripts/train_ppo_curriculum.py
```

**Note**: The training script automatically loads the previous phase's model when you specify `--phase` with a phase number > 1. For example, `--phase 3` will automatically load `ppo_phase2_final.zip` if it exists.

### Evaluate Trained Agent

```bash
# Full evaluation against all opponents
uv run python scripts/evaluate_agent.py --weights models/ppo_final.zip

# Test against specific opponent
uv run python scripts/evaluate_agent.py --weights models/ppo_final.zip --opponent reactive_ai
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

The agent progresses through 5 phases with graduated hit rewards and decaying hyperparameters. New models use a 256×256 policy network, longer rollouts, and a 0.995 discount factor.

| Phase | Opponent | Ball Speed | Steps | Target WR | LR | Hit Reward |
|-------|----------|------------|-------|-----------|----|------------|
| **1: Easy Wins** | SLOW_AI (40%) | 0.6x | 100K | 60% | 3e-4 | 0.5 |
| **2: Beginner** | BEGINNER_AI (55%) | 0.7x | 100K | 50% | 2.5e-4 | 0.3 |
| **3: Medium** | MEDIUM_AI (65%) | 0.8x | 150K | 40% | 1.5e-4 | 0.1 |
| **4: Competitive** | NORMAL_AI (70%) | 0.9x | 200K | 30% | 5e-5 | 0.05 |
| **5: Master** | EXPERT_AI (full prediction) | 1.0x | 300K | 20% | 3e-5 | 0.05 |

**Total: 850,000 base steps**. Phase 5 mixes 4 EXPERT, 2 REACTIVE, and 2 NORMAL environments to combine challenge with skill retention. Existing checkpoints remain loadable; new physics mean they should be re-evaluated before further training.

**Anti-regression**: Training automatically stops if the agent regresses on easy opponents (catastrophic forgetting detection).

### Graduated Reward Shaping

| Event | Reward | Purpose |
|-------|--------|---------|
| Score a point | +5 | Primary objective |
| Opponent scores | -5 | Penalty |
| Win game | +10 | Ultimate goal |
| Lose game | -10 | Strong negative signal |
| Hit ball | +0.5 → +0.05 | Graduated: bootstraps motor skills, then fades |
| Pressure opponent | +1.0 → +0.5 | Offensive shot placement shaping |
| Per step | -0.001 | Faster games |

**Key**: Hit rewards fade gradually across phases. Pressure shaping is paid once per return, not per step.

---

## 📊 Evaluation

### Historical PPO Results

These results came from the previous physics and are not benchmarks for the upgraded engine:

| Opponent | Win Rate | Description |
|----------|----------|-------------|
| SLOW_AI | 93% | Dominates |
| BEGINNER_AI | 90% | Strong |
| MEDIUM_AI | 77% | Good offensive play |
| NORMAL_AI | 43% | Competitive |
| REACTIVE_AI | 3% | Still challenging |

### Example Evaluation

```bash
$ uv run python scripts/evaluate_agent.py --weights models/ppo_final.zip --episodes 30
```

---

## 🎮 Environment Details

### Physics and Controls

- Swept collisions prevent fast balls from tunneling through paddles.
- Contact position sets the rebound angle; paddle motion adds bounded spin.
- Ball speed increases per return but remains capped, including its vertical component.
- Serves use playable random angles and a short pause in the graphical game.
- Use **UP/DOWN** or **W/S** to move. Run `uv run pong --difficulty normal` for an easier opponent.

### Observation Space (9 dimensions, normalized)

```
[ball_x, ball_y, ball_vx, ball_vy, player_y, opponent_y, ball_dist, player_score, opp_score]
```

All values normalized to [-1, 1] or [0, 1] for stable neural network training. Optional paddle-velocity, intercept, and relative-offset features can extend this vector for newly trained models.

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
| EXPERT_AI | 100% | 1px | Full wall prediction and attacking shot placement |

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
│   ├── CLI.md                   # Available uv run commands
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
model, passed = train_phase(
    phase=1,
    n_envs=16,           # More parallel environments
    save_dir="./my_models/",
    verbose=1
)
# `passed` is True if the phase's win-rate target was reached
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
from pong.env.pong_headless import register_headless_env

# Register "PongHeadless-v0" with gymnasium (not done automatically on import)
register_headless_env()

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

- **[CLI Commands](docs/CLI.md)** - Available `uv run` commands
- **[PPO Training Guide](docs/PPO_TRAINING.md)** - Detailed training methodology
- **[Environment Guide](learn/environment_guide.md)** - Environment API reference
- **[SB3 Tutorial](learn/stable_baselines3_tutorial.md)** - Stable-Baselines3 usage

---

## 🧪 Testing

```bash
# Test environment
uv run python pong/env/pong_headless.py

# Run test suite
uv run pytest tests/ -v

# Quick smoke test
uv run python -c "
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
| `ModuleNotFoundError: No module named 'pong'` | Use `uv run` to run commands (it sets up the path automatically) |
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
