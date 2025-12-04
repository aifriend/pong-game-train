# Changelog

All notable changes to the Pong RL Training Environment project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Critical: Curriculum phase advancement stalling** - Multiple bugs caused training to stall in Phase 2 for 900+ episodes:
  1. **Adaptive threshold reduction bug**: Logic used `if/if` instead of `if/elif`, causing 20% reduction at 4x episodes to never apply (2.5x condition always fired first). Fixed ordering with more conservative reductions (10%/15%/20% at 3x/5x/8x).
  2. **Consecutive counter reset bug**: Counter used per-episode metric values which are inherently noisy (hit_rate can be 0/0=0, 1/1=1.0, etc.), causing constant resets. Now uses 20-episode rolling average.
  3. **Hard reset on noisy metrics**: Single bad episode reset counter to 0. Now uses soft decay (decrement by 1) to preserve progress.

- **Critical: Premature Phase 3 advancement** - Agent advanced to Phase 3 with only 45% hit_rate, causing 0-5 losses every game (Score: -25). Added explicit hit_rate floor check (≥55%) when advancing FROM Phase 2 to Phase 3.

- **Critical: Replay buffer action indexing bug** - `Memory.sample_batch` and `PrioritizedMemory.sample_batch` were using `actions[sampled_indices]` instead of `actions[next_indices]`, causing each state to be paired with the *previous* action instead of the action that produced the transition. This corrupted TD updates and caused Phase 1 alignment and scores to regress during training. After fix: alignment improved from 0.40 to 0.71, scores from ~300 to ~3000.

### Changed
- **Curriculum variance thresholds significantly increased** - Phase 2-5 variance thresholds increased to realistic levels for noisy metrics:
  - Phase 2 (hit_rate): 0.20 → 0.35
  - Phase 3-4 (win_rate): 0.18 → 0.30
  - Phase 5 (win_rate): 0.18 → 0.28
- **Phase advancement thresholds adjusted**:
  - Phase 2: 0.65 → 0.60 (60% hit rate is solid)
  - Phase 5: 0.48 → 0.45 (more achievable vs reactive AI)
- **Stability episodes reduced** with rolling average:
  - Phase 2: 20 → 15
  - Phase 3: 30 → 20  
  - Phase 4: 30 → 25
  - Phase 5: 40 → 30
- **Minimum episode requirements reduced**:
  - Phase 3: 200 → 150
  - Phase 4: 250 → 200
  - Phase 5: 400 → 300
- **MIN_HIT_RATE_FLOOR reduced**: 0.70 → 0.55 (less restrictive for Phase 3+)

### Added
- Unit tests for curriculum fixes (`tests/test_curriculum.py`)
- Unit tests for replay buffer transition alignment (`tests/test_agent_memory.py`)

### Planned
- Multi-agent tournament mode
- Prioritized Experience Replay (PER) - *Already implemented*
- Curriculum learning with difficulty progression - *Already implemented*
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

