# Learning Resources

Educational materials for understanding the Pong AI training environment.

## 📚 Available Guides

### [Stable-Baselines3 Tutorial](stable_baselines3_tutorial.md)
Complete guide to using Stable-Baselines3 for Pong training, including PPO, DQN, and advanced features.

**Topics covered:**
- What is Stable-Baselines3?
- Core concepts and API
- PPO vs DQN algorithms
- Callbacks and monitoring
- Vectorized environments
- Custom policies

---

## 🎯 Quick Learning Path

### Beginner
1. Read [QUICKSTART.md](../QUICKSTART.md)
2. Run the basic training script
3. Understand observation and action spaces

### Intermediate
1. Read [PPO_TRAINING.md](../docs/PPO_TRAINING.md)
2. Understand the 3-phase curriculum
3. Experiment with hyperparameters

### Advanced
1. Read [stable_baselines3_tutorial.md](stable_baselines3_tutorial.md)
2. Implement custom reward wrappers
3. Design your own curriculum

---

## 🔬 Key Concepts

### Reinforcement Learning Basics

- **Agent**: The AI player learning to play Pong
- **Environment**: The Pong game that the agent interacts with
- **Observation**: The 9-dimensional state vector the agent sees
- **Action**: Move up, down, or stay
- **Reward**: +10 for scoring, -10 for opponent scoring, +20 for winning

### PPO (Proximal Policy Optimization)

- **On-policy** algorithm (learns from current policy)
- **Actor-Critic** architecture (policy + value networks)
- **Clipped objective** for stable updates
- **Best for**: Sparse rewards, game environments

### Win-Focused Training

The key insight: **reward scoring, not rallying**

Previous approaches rewarded hitting the ball, which made agents learn to rally indefinitely. The PPO approach only rewards:
- Scoring points
- Winning games
- Offensive positioning

This forces the agent to learn aggressive, scoring-focused strategies.

---

## 📖 Additional Resources

### External Resources

- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [PPO Paper](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [OpenAI Spinning Up](https://spinningup.openai.com/)

### In This Repository

- [README.md](../README.md) - Project overview
- [QUICKSTART.md](../QUICKSTART.md) - Fast setup guide
- [PPO_TRAINING.md](../docs/PPO_TRAINING.md) - Training methodology
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines

---

## 🎓 Learning by Example

### Example 1: Basic Training

```python
from stable_baselines3 import PPO
from pong.env.wrappers import make_ppo_env
from pong.env.pong_headless import OpponentType

# Create environment
env = make_ppo_env(OpponentType.SLOW_AI)

# Create PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train for 50K steps
model.learn(total_timesteps=50000)

# Save
model.save("my_first_agent")
```

### Example 2: Evaluation

```python
from stable_baselines3 import PPO
from pong.env.pong_headless import PongHeadlessEnv, OpponentType

# Load trained model
model = PPO.load("my_first_agent")

# Create test environment
env = PongHeadlessEnv(opponent_type=OpponentType.NORMAL_AI)

# Play 10 games
wins = 0
for episode in range(10):
    obs, _ = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    if info["player_score"] > info["opponent_score"]:
        wins += 1
    
    print(f"Game {episode+1}: {info['player_score']}-{info['opponent_score']}")

print(f"\nWin Rate: {wins}/10 = {wins*10}%")
```

---

## 💡 Tips for Success

1. **Start simple**: Use Phase 1 only to understand the system
2. **Monitor training**: Always use TensorBoard to watch learning
3. **Be patient**: Early training shows 0% win rate - this is normal
4. **Test frequently**: Evaluate after each phase to track progress
5. **Experiment**: Try different hyperparameters and opponents

---

## ❓ FAQ

**Q: How long does training take?**  
A: ~4 hours for full curriculum (8 parallel environments)

**Q: Can I train on CPU?**  
A: Yes! PPO works well on CPU. GPU provides minimal benefit for this small model.

**Q: Why is win rate 0% at the start?**  
A: The agent starts with random actions. It needs ~20K steps to learn basic skills.

**Q: Can I resume training?**  
A: Yes! Use `--phase 2` or `--phase 3` to continue from saved models.

**Q: How do I know if training is working?**  
A: Check TensorBoard. `rollout/ep_rew_mean` should increase from -70 toward +30 over time.

---

## 🎯 Next Steps

After your first successful training:

1. **Read** [PPO_TRAINING.md](../docs/PPO_TRAINING.md) for deeper understanding
2. **Experiment** with different opponents and reward structures
3. **Contribute** improvements back to the project
4. **Share** your trained agents and results!

---

**Ready? Let's train!** 🏓

```bash
uv run python scripts/train_ppo_curriculum.py
```

