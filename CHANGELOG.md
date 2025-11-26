# Changelog

All notable changes to the Pong RL Training Environment project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Multi-agent tournament mode
- Prioritized Experience Replay (PER)
- Curriculum learning with difficulty progression
- Docker containerization
- Web-based training dashboard

## [1.0.0] - 2025-01-XX

### Added
- **Dual Environment Support**
  - `PongEnv`: Full pygame-based environment with graphics and audio
  - `PongHeadlessEnv`: Pure Python environment for ultra-fast headless training
  - Self-play training mode (agent vs agent)

- **Advanced RL Algorithms**
  - Custom Double DQN implementation with PyTorch
  - Stable-Baselines3 integration (PPO and DQN)
  - TensorBoard integration for real-time monitoring

- **Training Features**
  - Optimized hyperparameters for fast learning
  - Normalized observations (all inputs scaled to [0,1] or [-1,1])
  - Intelligent reward shaping
  - Efficient numpy-based experience replay buffer
  - Checkpoint system (saves every 100 episodes)

- **Interactive Testing**
  - Human vs Agent mode (`play_against_agent.py`)
  - Progressive checkpoint testing
  - Configurable game modes (ball speed, max score, episode length)

- **Documentation**
  - Comprehensive README with examples
  - Environment specification guide
  - Stable-Baselines3 tutorial
  - Training configuration guide
  - API documentation

- **Testing**
  - Gym integration tests
  - Environment validation tests
  - Project structure tests

- **Configuration Management**
  - Multiple pre-configured training modes (DEFAULT, FAST, DEBUG, PRODUCTION)
  - Easy-to-customize config system
  - Environment variable support

### Project Structure
```
pong/               - Core game package
  env/              - Gymnasium environments
  game/             - Game components (ball, paddle, manager)
  constants.py      - Game constants

trainer/            - Custom DQN implementation
  main.py           - Training entry point
  the_agent_pytorch.py - Double DQN agent
  agent_memory.py   - Experience replay buffer
  environment.py    - Training environment wrapper
  config.py         - Configuration management

scripts/            - Utility scripts
  train_pong_agent.py     - Stable-Baselines3 training
  play_against_agent.py   - Human vs AI testing
  play.py                 - Manual game play

tests/              - Test suite
learn/              - Tutorials and guides
docs/               - API documentation
resources/          - Game assets (images, sounds, fonts)
```

### Performance
- Headless training: ~528 steps/sec
- Visual training: ~50-100 steps/sec
- Observation normalization: 2-3x faster convergence
- Optimized memory: 5K starting experiences (vs 50K baseline)

### Dependencies
- Python 3.8+
- pygame ~2.1.2
- gymnasium >=0.29.0
- numpy >=1.21.0
- stable-baselines3 >=2.0.0
- matplotlib >=3.5.0
- torch >=2.0.0
- tensorboard >=2.10.0

---

## Version History

### Version Naming Convention
- **Major version** (X.0.0): Breaking changes, major new features
- **Minor version** (1.X.0): New features, backward compatible
- **Patch version** (1.0.X): Bug fixes, small improvements

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

