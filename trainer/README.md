# Trainer Module (PyTorch DQN)

This directory contains the custom Deep Q-Network (DQN) trainer that now runs **entirely on PyTorch**. It talks directly to the `Pong-v0` Gymnasium environment and lets you train, debug, and extend a bespoke agent without touching TensorFlow.

## What's Inside

| File | Purpose |
| --- | --- |
| `main.py` | Entry point for training. Creates the agent, environment, prints stats, and orchestrates episodes. |
| `the_agent_pytorch.py` | PyTorch implementation of the DQN agent (networks, replay, learning loop, checkpoints). |
| `agent_memory.py` | Simple experience replay buffer used by the agent. |
| `environment.py` | Thin wrapper around Gymnasium to register `Pong-v0`, reset episodes, and step while syncing memory. |
| `debug.py` | Optional helper for stepping through the agent/environment visually. |
| `preprocess_frame.py` | Legacy helper kept for reference; not used with the current vector observations. |
| `__init__.py` | Exposes `Agent` and `Memory` so other modules can import them directly. |

## Running Training

```bash
source .venv/bin/activate
python trainer/main.py
```

During training you will see per-episode stats (score, duration, loss, speed). Press `Ctrl+C` to stop; weights are saved automatically.

## Saved Weights & Automatic Resume

The PyTorch agent stores checkpoints as `.pth` files:

- `checkpoints/checkpoint_episode_N.pth` is saved every 100 episodes with full training state.
- `final_weights.pth` is written when `main.py` finishes or you interrupt training.

### Automatic Checkpoint Resume

Training automatically resumes from the latest checkpoint when you restart `main.py`:

1. The script scans `checkpoints/` for checkpoint files
2. Loads the checkpoint with the highest episode number
3. Restores full training state including:
   - Model weights and optimizer state
   - Episode counter and curriculum phase
   - Training statistics (scores history, max score)
   - Plotting history for continuous progress graphs
   - Elapsed training time

Simply run `python trainer/main.py` and training will continue from where it left off.

### Manual Weight Loading

To manually load specific weights:

```python
from trainer import Agent

agent = Agent(...)
training_state = agent.load_weights('checkpoints/checkpoint_episode_1000.pth')
# training_state contains: episode, current_phase, scores_history, max_score, etc.
```

## Hyperparameters

Key values live near the top of `main.py`:

```python
POSSIBLE_ACTIONS = [0, 1, 2]  # stay, up, down
STARTING_MEM_LEN = 5000
MAX_MEM_LEN = 100000
LEARN_RATE = 0.0005
OBSERVATION_DIM = 9
MAX_EPISODES = 100000
RENDER_MODE = None  # set to 'human' to enable window
```

- Adjust `RENDER_MODE` if you want visual training.
- `STARTING_MEM_LEN` gates when learning begins (agent collects experiences first).
- Exploration is handled by noisy networks (no epsilon-greedy).

## Dependencies

The trainer relies on packages already listed in `requirements.txt`:

- `torch` (brought in via Stable-Baselines3)
- `gymnasium`
- `numpy`
- `pygame` (for rendering through the custom environment)

Ensure the virtual environment is active so these imports work.

## Tips

- Keep `max_score` low (default 5) for faster iterations.
- Use `python trainer/debug.py` if you need a quick visual sanity check.
- Noisy networks handle exploration automatically (no epsilon tuning needed).
- Profile training with `tensorboard --logdir ./tensorboard_dqn/` for detailed metrics.

No legacy TensorFlow paths remain—this folder is purely PyTorch. If you add new tools or scripts here, keep this README updated so everything under `trainer/` stays self-documented.
