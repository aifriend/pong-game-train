"""
Debug script for visualizing agent behavior and memory states.
"""

import numpy as np
from trainer import environment
from trainer.the_agent_pytorch import Agent
import sys
from pathlib import Path

# Add project root to path so imports work when running directly
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


ENV_NAME = "Pong-v0"
POSSIBLE_ACTIONS = [0, 1, 2]  # 0: stay, 1: up, 2: down
OBSERVATION_DIM = 9


def debug_episode():
    """Run a debug episode with rendering."""
    print("🐛 Debug Mode - Running Pong Episode")
    print("=" * 50)

    # Create agent (noisy networks handle exploration)
    agent = Agent(
        possible_actions=POSSIBLE_ACTIONS,
        starting_mem_len=50,
        max_mem_len=100000,
        learn_rate=0.001,
        observation_dim=OBSERVATION_DIM,
        debug=True,
        learn_every=8,
        batch_size=128,
        target_update_freq=5000,
    )

    # Create environment with human rendering
    env = environment.make_env(ENV_NAME, agent, render_mode="human")

    # Play one episode
    print("\nPlaying episode with rendering...")
    score = environment.play_episode(env, agent, debug=True)

    print(f"\nEpisode completed!")
    print(f"Final score: {score:.2f}")
    print(f"Total steps: {agent.total_timesteps}")
    print(f"Memory size: {len(agent.memory.frames)}")

    env.close()

    # Print memory statistics
    print("\n" + "=" * 50)
    print("Memory Statistics:")
    print(f"Total experiences: {len(agent.memory.frames)}")
    print(f"Average reward: {np.mean(agent.memory.rewards):.4f}")
    print(f"Total rewards: {sum(agent.memory.rewards):.2f}")
    print(f"Terminal states: {sum(agent.memory.done_flags)}")

    # Show first few experiences
    print("\nFirst 5 experiences:")
    for i in range(min(5, len(agent.memory.frames))):
        print(f"  Step {i}:")
        print(f"    Observation: {agent.memory.frames[i]}")
        print(f"    Action: {agent.memory.actions[i]}")
        print(f"    Reward: {agent.memory.rewards[i]:.4f}")
        print(f"    Done: {agent.memory.done_flags[i]}")


if __name__ == "__main__":
    debug_episode()
