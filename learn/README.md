# Learning Resources

Welcome to the Pong RL learning resources! This directory contains comprehensive guides and tutorials for working with reinforcement learning in the Pong environment.

## 📚 Available Guides

### 1. [Stable-Baselines3 Tutorial](./stable_baselines3_tutorial.md)
Complete guide to using Stable-Baselines3 with the Pong environment. Learn how to:
- Set up and train RL agents
- Use PPO and DQN algorithms
- Implement callbacks and monitoring
- Best practices and common patterns

### 2. [Environment Guide](./environment_guide.md)
Detailed documentation about the Pong Gym environment:
- Environment specifications
- Observation and action spaces
- Reward system
- Configuration options
- Usage examples

### 3. [Legacy Trainer Guide](./legacy_trainer_guide.md)
Guide to the custom DQN implementation:
- Understanding DQN from scratch
- Custom network architectures
- Educational implementation details
- When to use custom vs library implementations

## 🚀 Quick Start

### For Beginners
1. Start with [Environment Guide](./environment_guide.md) to understand the Pong environment
2. Follow [Stable-Baselines3 Tutorial](./stable_baselines3_tutorial.md) for practical training
3. Explore [Legacy Trainer Guide](./legacy_trainer_guide.md) to understand the internals

### For Experienced Users
- Jump to [Stable-Baselines3 Tutorial](./stable_baselines3_tutorial.md) for advanced features
- Check [Environment Guide](./environment_guide.md) for configuration options
- Review [Legacy Trainer Guide](./legacy_trainer_guide.md) for custom implementations

## 📖 Additional Resources

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/)

## 🎯 Learning Path

```
1. Understand the Environment
   └─> Read environment_guide.md
   
2. Learn Stable-Baselines3
   └─> Follow stable_baselines3_tutorial.md
   
3. Understand Internals (Optional)
   └─> Study legacy_trainer_guide.md
   
4. Experiment and Practice
   └─> Run scripts/train_pong_agent.py
   └─> Modify hyperparameters
   └─> Try different algorithms
```

## 💡 Tips

- **Start Simple**: Begin with default hyperparameters
- **Monitor Training**: Always use TensorBoard to track progress
- **Experiment**: Try different algorithms and configurations
- **Read Code**: The example scripts are well-commented
- **Ask Questions**: Check documentation and examples

## 🔧 Practical Examples

All practical examples can be found in:
- `scripts/train_pong_agent.py` - Main training script
- `scripts/play.py` - Play the game manually
- `trainer/main.py` - Custom DQN implementation
- `tests/test_gym_integration.py` - Environment tests

Happy Learning! 🎮🤖

