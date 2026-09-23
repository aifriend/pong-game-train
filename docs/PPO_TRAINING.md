# PPO Training Guide

This document describes the PPO-based training approach for the Pong AI agent.

## Overview

The PPO (Proximal Policy Optimization) training approach was developed to address limitations of the original DQN training method, specifically the agent's tendency to learn defensive/rallying strategies instead of winning games.

### Why PPO?

| Aspect | DQN | PPO |
|--------|-----|-----|
| Learning Type | Off-policy | On-policy |
| Sparse Rewards | Struggles | Handles well |
| Training Stability | Requires careful tuning | More stable |
| Credit Assignment | Shorter horizon | Better long-term |
| Parallel Training | Limited | Native support |

### Problem with Previous Approach

The original DQN agent plateaued after 300+ hours of training with 0% win rate because:

1. **Reward Structure**: Hit rewards (+1.2) were frequent and safe, while scoring rewards (+/-12) were rare
2. **Behavior Learned**: Agent optimized for rallying (safe, consistent reward) over scoring (risky, sparse reward)
3. **Off-Policy Limitations**: DQN's experience replay struggled with changing strategies

## Training Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Training Pipeline                      │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  PongHeadlessEnv                                         │
│        ↓                                                  │
│  WinFocusedRewardWrapper (pressure + graduated hit)      │
│        ↓                                                  │
│  EpisodeStatsWrapper (tracking)                          │
│        ↓                                                  │
│  DummyVecEnv (8 parallel environments)                   │
│        ↓                                                  │
│  PPO Agent (MlpPolicy, 136K params)                      │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

## 5-Phase Curriculum

The curriculum uses **graduated difficulty** with **small increments (~10% opponent speed per phase)**, **decaying hit rewards** (0.5 → 0.05), **decaying hyperparameters** (LR + entropy), and **strong offensive pressure shaping** to prevent catastrophic forgetting.

### Phase 1: Easy Wins (100K steps)

- **Opponent**: SLOW_AI (40% speed, 35px dead zone)
- **Ball Speed**: 0.6x
- **LR**: 3e-4, **Entropy**: 0.05
- **Rewards**: hit=0.5, pressure=1.0, tracking=0.02
- **Goal**: Bootstrap motor skills - learn to hit ball and score
- **Target Win Rate**: 60%

High hit reward teaches the agent to make contact with the ball. Pressure shaping, paid once per return, guides offensive placement.

### Phase 2: Beginner (100K steps)

- **Opponent**: BEGINNER_AI (55% speed, 28px dead zone)
- **Ball Speed**: 0.7x
- **LR**: 2.5e-4, **Entropy**: 0.04
- **Rewards**: hit=0.3, pressure=1.0, tracking=0.02
- **Goal**: Transfer skills to faster opponent
- **Target Win Rate**: 50%

Reduced hit reward starts transitioning the agent from contact-seeking to score-seeking behavior.

### Phase 3: Medium (150K steps)

- **Opponent**: MEDIUM_AI (65% speed, 22px dead zone)
- **Ball Speed**: 0.8x
- **LR**: 1.5e-4, **Entropy**: 0.03
- **Rewards**: hit=0.1, pressure=0.8, tracking=0.015
- **Goal**: Learn offensive shot placement against competent opponent
- **Target Win Rate**: 40%

Low hit reward. Agent must rely more on scoring and pressure shaping for positive reward.

### Phase 4: Competitive (200K steps)

- **Opponent**: NORMAL_AI (70% speed, 20px dead zone)
- **Ball Speed**: 0.9x
- **LR**: 5e-5, **Entropy**: 0.015
- **Rewards**: hit=0.05, pressure=0.65, tracking=0.01
- **Goal**: Beat a fast opponent with near-full ball speed
- **Target Win Rate**: 30%

Very low LR prevents catastrophic forgetting. Minimal hit reward keeps value function stable during the transition.

### Phase 5: Master (200K steps)

- **Opponent**: REACTIVE_AI (70% speed, 18px dead zone + late ball prediction)
- **Ball Speed**: 1.0x
- **LR**: 3e-5, **Entropy**: 0.01
- **Rewards**: hit=0.05, pressure=0.5, tracking=0.01
- **Goal**: Master the predictive AI at full speed
- **Target Win Rate**: 20%
- **Mixed Training**: 4 REACTIVE + 2 NORMAL + 2 MEDIUM environments

The REACTIVE_AI uses **late prediction** - it only starts predicting ball trajectory in the last 30% of the court. This gives the agent's angled shots time to create scoring opportunities before the opponent reacts.

**Mixed-opponent training** prevents catastrophic forgetting by maintaining skill against easier opponents while learning to beat the hardest one.

### Anti-Regression Protection

A regression detection callback evaluates against SLOW_AI periodically:
- Phases 1-3: every 50K steps
- Phases 4-5: every 25K steps (more sensitive to forgetting)

If win rate drops below 50% against SLOW_AI, training stops immediately.

## Rewards (Graduated Offensive Focus)

The `WinFocusedRewardWrapper` provides **graduated reward shaping** that transitions from motor skill bootstrap to pure win-focus across phases:

### Hit Rewards (graduated across phases)
- Phase 1: +0.5 (bootstrap motor skills)
- Phase 2: +0.3 (reduced)
- Phase 3: +0.1 (minimal)
- Phase 4-5: +0.05 (prevents value function collapse)

### Offensive Reward (paid once, when the player returns the ball)
- **Pressure shaping**: +1.0 → +0.5 (decaying) when the return heads where opponent ISN'T
  - Paid ONCE per return, not on every step the ball travels toward the opponent
  - Predicts ball y-intersection at opponent's x using wall-bounce reflection
  - Rewards proportional to distance from opponent paddle

### Defensive Rewards (when ball moving toward player)
- **Tracking**: +0.02 → +0.01 when paddle is aligned with ball

### Sparse Rewards (on events)
- Score a point: +5.0
- Opponent scores: -5.0
- Win the game: +10.0
- Lose the game: -10.0

### Penalties
- Per step: -0.001 (encourages faster games)

**Key design**: Hit rewards fade gradually so the agent always has *some* learning signal. Pressure shaping is strong (a perfectly placed return pays the full scale, 20% of the 5.0 point reward at 1.0) to overcome the sparse nature of scoring rewards.

## Opponent Configuration

| Type | Speed | Dead Zone | Special Behavior |
|------|-------|-----------|------------------|
| SLOW_AI | 40% | 35px | Slow reactions, large blind spot |
| BEGINNER_AI | 55% | 28px | Moderate challenge |
| MEDIUM_AI | 65% | 22px | Competent opponent |
| NORMAL_AI | 70% | 20px | Fast and competitive |
| REACTIVE_AI | 70% | 18px | Late ball trajectory prediction (last 30% of court) |

### REACTIVE_AI Late Prediction

The REACTIVE_AI opponent predicts where the ball will arrive, but only activates this prediction when the ball enters the last 30% of the court (near the opponent side). This design:

1. Gives the agent's angled shots time to create wide angles before the opponent reacts
2. Makes the opponent beatable (~15% pre-training win rate) while still being challenging
3. Prevents the "speed cliff" problem where prediction + high speed makes opponents physically unbeatable

**Physics constraint discovered during development**: At speeds > 0.70, the AI paddle can cover the full court height during ball crossing time, making it impossible to score regardless of shot angle.

## Usage

### Quick Start

```bash
# Run full curriculum (all 5 phases)
uv run python scripts/train_ppo_curriculum.py
```

### Training Options

```bash
# Start from specific phase
python scripts/train_ppo_curriculum.py --phase 2

# Train single phase only
python scripts/train_ppo_curriculum.py --phase 1 --single-phase

# Custom parallel environments
python scripts/train_ppo_curriculum.py --envs 4

# Custom save directory
python scripts/train_ppo_curriculum.py --save-dir ./my_models/
```

### Evaluate Trained Model

```bash
# Evaluate SB3 model
python scripts/evaluate_agent.py --weights models/ppo_final.zip --episodes 50

# Evaluate against specific opponent
python scripts/evaluate_agent.py --weights models/ppo_final.zip --opponent reactive_ai
```

### Monitor Training

```bash
# TensorBoard
tensorboard --logdir ./tensorboard/
```

## Expected Results

### Training Time

| Phase | Steps | LR | Approximate Time |
|-------|-------|-----|------------------|
| 1: Easy Wins | 100K | 3e-4 | ~1 min |
| 2: Beginner | 100K | 2.5e-4 | ~1 min |
| 3: Medium | 150K | 1.5e-4 | ~2 min |
| 4: Competitive | 200K | 5e-5 | ~3 min |
| 5: Master | 200K | 3e-5 | ~8 min |
| **Total** | **750K** | - | **~15 min** |

*Base step budgets. Each phase has a 5x multiplier for max attempts. Times based on 8 parallel environments on M1 Mac.*

### Win Rate Targets

| Opponent | Target | Phase Completed |
|----------|--------|-----------------|
| SLOW_AI | 60%+ | After Phase 1 |
| BEGINNER_AI | 50%+ | After Phase 2 |
| MEDIUM_AI | 40%+ | After Phase 3 |
| NORMAL_AI | 30%+ | After Phase 4 |
| REACTIVE_AI | 20%+ | After Phase 5 |

### Best Achieved Results

| Opponent | Win Rate | Notes |
|----------|----------|-------|
| SLOW_AI | 93% | Dominates |
| BEGINNER_AI | 90% | Strong |
| MEDIUM_AI | 77% | Good offensive play |
| NORMAL_AI | 43% | Competitive |
| REACTIVE_AI | 3% | Still challenging |

## File Structure

```
pong/
├── env/
│   ├── pong_headless.py
│   └── wrappers.py         # WinFocusedRewardWrapper (pressure shaping)
├── scripts/
│   ├── train_ppo_curriculum.py  # Main PPO training
│   └── evaluate_agent.py        # Updated for SB3 support
└── docs/
    └── PPO_TRAINING.md          # This file
```

## Hyperparameters

### PPO Configuration

```python
PPO(
    policy="MlpPolicy",
    learning_rate=3e-4,     # Phase 1: 3e-4 → Phase 5: 3e-5
    n_steps=1024,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.05,          # Phase 1: 0.05 → Phase 5: 0.01
    vf_coef=0.5,
    max_grad_norm=0.5,
)
```

### Reward Configuration (Phase 1 example)

```python
WinFocusedRewardWrapper(
    point_reward=5.0,
    win_bonus=10.0,
    hit_reward=0.5,         # Decays: 0.5 → 0.3 → 0.1 → 0.05 → 0.05
    tracking_reward=0.02,   # Decays: 0.02 → 0.01
    pressure_scale=1.0,     # Decays: 1.0 → 0.5
    step_penalty=0.001,
)
```

### Environment Configuration

- **Parallel Environments**: DummyVecEnv with 8 envs (default)
- **Max Score**: 3 (first to 3; measured so games conclude in ~2,800 steps)
- **Max Steps**: 5000 (prevents infinite episodes)
- **Ball acceleration**: +25% horizontal speed per paddle hit, capped at 4x launch speed (classic Pong mechanic)
- **Serve speed**: base 6.0 px/step; phase multipliers 0.6x → 1.0x scale it proportionally
- **Phase 5**: Mixed-opponent DummyVecEnv (4 REACTIVE + 2 NORMAL + 2 MEDIUM)

### Phase Gating

Phase advancement uses **deterministic evaluation** to ensure stable policy assessment:
- 100 evaluation episodes per check, with greedy (deterministic) actions
- 100 episodes for final evaluation as well, since it gates the same decision
- 100 episodes give a standard error near 4.6 percentage points at a 30% win
  rate, against 10.2 points with 20 episodes
- Must hit target 2 consecutive times for early advancement
- 5x step budget multiplier before giving up on a phase

## Troubleshooting

### Low Win Rate in Phase 1

If win rate is below 60% after Phase 1:
1. Increase Phase 1 max timesteps multiplier
2. Verify the wrapper is applied correctly
3. Check that hit_reward=0.5 is being passed to the environment

### Training Too Slow

1. Reduce `n_envs` if running out of memory
2. Use `--envs 4` for lower-spec machines
3. Check CPU usage - should be near 100% per core

### Agent Not Improving

1. Check TensorBoard for learning curves
2. Verify rewards are non-zero in logs
3. Ensure pressure shaping is active (ball_vx < 0 periods)

### Regression Detected (Training Stops Early)

This means the agent is forgetting basic skills while learning harder ones:
1. The model before regression is automatically saved
2. Try reducing the learning rate for the failing phase
3. Consider adding a mixed-opponent approach (like Phase 5 uses)

### 0% Win Rate Against REACTIVE_AI

REACTIVE_AI is the hardest opponent due to ball prediction:
- The late-prediction design (last 30% of court) makes it beatable but challenging
- Mixed-opponent training in Phase 5 prevents catastrophic forgetting
- Even partial win rates (3-10%) indicate the agent is learning offensive play

## Comparison with Previous Approach

| Metric | DQN (312+ hours) | PPO (15 min) |
|--------|------------------|--------------|
| Training Time | 312+ hours | ~15 minutes |
| Win Rate vs NORMAL_AI | 0% | 43% |
| Win Rate vs REACTIVE_AI | 0% | 3% |
| Approach | Defensive/Rally | Offensive/Score |
| Reward Focus | Hit ball | Pressure opponent |

## References

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PPO Paper (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [OpenAI Spinning Up - PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
