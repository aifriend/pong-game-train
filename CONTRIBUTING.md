# Contributing to Pong RL Training Environment

First off, thank you for considering contributing to Pong RL Training Environment! It's people like you that make this project such a great tool for the RL community.

## 🎯 Ways to Contribute

### 🐛 Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** (code snippets, terminal output)
- **Describe the behavior you observed** and what you expected
- **Include your environment details** (OS, Python version, package versions)

### ✨ Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description** of the suggested enhancement
- **Explain why this enhancement would be useful**
- **List some examples** of how it would be used

### 🔧 Pull Requests

1. **Fork the repository** and create your branch from `master`
2. **Follow the development setup** instructions below
3. **Make your changes** and ensure they follow the code style
4. **Add tests** if applicable
5. **Update documentation** if needed
6. **Ensure all tests pass**
7. **Submit your pull request**

## 🛠️ Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Setup Steps

```bash
# Clone your fork
git clone https://github.com/your-username/pong-rl-env.git
cd pong-rl-env

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_gym_integration.py -v

# Run with coverage
python -m pytest tests/ --cov=pong --cov-report=html
```

## 📝 Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking (optional)

```bash
# Format code
black pong/ trainer/ scripts/ tests/

# Check linting
flake8 pong/ trainer/ scripts/ tests/

# Type checking (optional)
mypy pong/
```

### Code Style Guidelines

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use meaningful variable and function names
- Write docstrings for all public functions and classes
- Keep functions focused and small
- Add type hints where possible

## 📚 Documentation

- Update README.md if adding new features or changing behavior
- Update docstrings for modified functions/classes
- Add examples in `learn/` directory for new features
- Update `docs/` if changing API or environment specifications

## 🎯 Areas We Need Help

Here are some areas where contributions would be especially valuable:

### High Priority
- [ ] Additional unit tests and integration tests
- [ ] Performance optimizations
- [ ] Documentation improvements and tutorials
- [ ] Bug fixes and issue resolution

### Medium Priority
- [ ] New RL algorithm implementations (A3C, SAC, TD3)
- [ ] Visualization tools and dashboards
- [ ] Multi-agent training modes
- [ ] Curriculum learning implementation

### Nice to Have
- [ ] Docker containerization
- [ ] CI/CD pipeline setup
- [ ] Pre-trained model zoo
- [ ] Web-based demo interface

## 🔄 Development Workflow

1. **Create an issue** describing what you plan to work on
2. **Wait for feedback** from maintainers (for large changes)
3. **Fork and create a branch** with a descriptive name:
   - Feature: `feature/add-ppo-algorithm`
   - Bug fix: `fix/memory-leak-issue`
   - Documentation: `docs/update-installation-guide`
4. **Make your changes** following the code style
5. **Write/update tests** to cover your changes
6. **Run tests locally** to ensure everything passes
7. **Commit with clear messages**:
   ```
   Add PPO algorithm implementation
   
   - Implement PPO with clipped surrogate objective
   - Add configuration options for PPO hyperparameters
   - Include unit tests for PPO agent
   - Update documentation with PPO usage examples
   ```
8. **Push to your fork** and create a pull request
9. **Respond to feedback** and make requested changes
10. **Celebrate** when your PR is merged! 🎉

## 🤝 Code of Conduct

### Our Pledge

We are committed to making participation in this project a harassment-free experience for everyone, regardless of:
- Age, body size, disability, ethnicity
- Gender identity and expression
- Level of experience
- Nationality, personal appearance, race
- Religion, sexual identity and orientation

### Our Standards

**Positive behavior includes:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what's best for the community
- Showing empathy towards other community members

**Unacceptable behavior includes:**
- Trolling, insulting/derogatory comments, personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct inappropriate in a professional setting

### Enforcement

Project maintainers are responsible for clarifying standards and will take appropriate action in response to unacceptable behavior.

## 📞 Getting Help

- 📖 Check the [documentation](docs/) and [tutorials](learn/)
- 💬 Open a [discussion](https://github.com/your-username/pong-rl-env/discussions)
- 🐛 Search [existing issues](https://github.com/your-username/pong-rl-env/issues)
- 📧 Contact maintainers (for security issues)

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for their contributions
- GitHub contributors page

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Pong RL Training Environment! 🎮🤖

