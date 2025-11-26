"""
Main training script for Double DQN agent on Pong environment.
Features: Self-play, TensorBoard logging, optimized hyperparameters.
"""

import numpy as np
from collections import deque
import time
import matplotlib.pyplot as plt
from trainer import environment
from trainer import Agent
import sys
import os
from pathlib import Path

# Add project root to path so imports work when running directly
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# Environment configuration
ENV_NAME = "Pong-v0"  # Custom Pong environment
RENDER_MODE = None  # Set to 'human' for visualization, None for faster training

# Agent hyperparameters (OPTIMIZED v2 - improved learning signal)
POSSIBLE_ACTIONS = [0, 1, 2]  # 0: stay, 1: up, 2: down
STARTING_MEM_LEN = 10000  # Increased for more diverse initial experiences
MAX_MEM_LEN = 100000  # More recent experiences
LEARN_RATE = (
    0.001  # Doubled for faster learning (low loss indicated need for more aggressive learning)
)
OBSERVATION_DIM = 9  # Pong-v0 observation space dimension
BATCH_SIZE = 128  # Doubled for more stable gradients

# Training configuration
MAX_EPISODES = 100000
SAVE_INTERVAL = 50  # Save plot every N episodes
CHECKPOINT_INTERVAL = 100  # Save model checkpoint every N episodes
SELF_PLAY = True  # Enable self-play (agent vs agent) training
TENSORBOARD_LOG = "./tensorboard_dqn/"  # TensorBoard log directory


def main():
    """Main training loop with Double DQN and TensorBoard logging."""
    print("🏓 Double DQN Training on Pong Environment")
    print("=" * 50)

    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)

    # Create agent with TensorBoard logging
    agent = Agent(
        possible_actions=POSSIBLE_ACTIONS,
        starting_mem_len=STARTING_MEM_LEN,
        max_mem_len=MAX_MEM_LEN,
        learn_rate=LEARN_RATE,
        observation_dim=OBSERVATION_DIM,
        tensorboard_log=TENSORBOARD_LOG,
        learn_every=8,  # Learn every 8 steps (more diverse experience per learn)
        batch_size=BATCH_SIZE,
        target_update_freq=5000,  # Less frequent target updates for stability
    )

    # Curriculum learning: create environments for different phases
    # Phase 1: vs simple AI (episodes 0-1000)
    # Phase 2: mixed 50/50 (episodes 1000-3000)
    # Phase 3: full self-play (episodes 3000+)
    env_ai = environment.make_env(ENV_NAME, agent, render_mode=RENDER_MODE, self_play=False)
    env_selfplay = environment.make_env(ENV_NAME, agent, render_mode=RENDER_MODE, self_play=True)

    # Curriculum thresholds
    PHASE1_END = 1000  # End of AI-only phase
    PHASE2_END = 3000  # End of mixed phase

    def get_env_for_episode(episode):
        """Get appropriate environment based on curriculum phase."""
        if episode < PHASE1_END:
            return env_ai, "AI"
        elif episode < PHASE2_END:
            # Mixed phase: 50% AI, 50% self-play
            if episode % 2 == 0:
                return env_ai, "AI"
            else:
                return env_selfplay, "Self-play"
        else:
            return env_selfplay, "Self-play"

    # Training statistics
    last_100_avg = []
    scores = deque(maxlen=100)
    max_score = float("-inf")
    total_start_time = time.time()
    current_phase = "AI"

    # Load weights if resuming training
    # Uncomment to load pre-trained weights:
    # if os.path.exists('recent_weights.pth'):
    #     agent.load_weights('recent_weights.pth')
    #     print("Loaded pre-trained weights")

    print(f"\nStarting training for {MAX_EPISODES} episodes...")
    print(f"Environment: {ENV_NAME}")
    print(f"Curriculum Learning Enabled:")
    print(f"  Phase 1 (0-{PHASE1_END}): vs Simple AI")
    print(f"  Phase 2 ({PHASE1_END}-{PHASE2_END}): Mixed (50% AI, 50% Self-play)")
    print(f"  Phase 3 ({PHASE2_END}+): Full Self-play")
    print(f"Render mode: {RENDER_MODE}")
    print(f"Learning rate: {LEARN_RATE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Starting memory: {STARTING_MEM_LEN}")
    print(f"Checkpoints saved every {CHECKPOINT_INTERVAL} episodes")
    print(f"TensorBoard: tensorboard --logdir {TENSORBOARD_LOG}")
    print("-" * 50)

    try:
        for episode in range(MAX_EPISODES):
            timesteps_before = agent.total_timesteps
            episode_start_time = time.time()

            # Curriculum learning: select environment based on phase
            env, phase = get_env_for_episode(episode)
            if phase != current_phase:
                print(f"\n🎓 Curriculum phase change: {current_phase} → {phase}")
                current_phase = phase

            # Play one episode
            score = environment.play_episode(env, agent, debug=False)

            # Update statistics
            episode_length = agent.total_timesteps - timesteps_before
            scores.append(score)
            if score > max_score:
                max_score = score

            # Log episode to TensorBoard
            agent.log_episode(episode, score, episode_length)

            # Print episode statistics (less verbose for faster training)
            episode_duration = time.time() - episode_start_time

            if episode % 10 == 0:  # Print every 10 episodes
                ep_speed = 1.0 / episode_duration if episode_duration > 0 else 0
                learns_per_ep = agent.learns / max(1, episode + 1)
                print(
                    f"Ep {episode:5d} | Steps: {episode_length:4d} | "
                    f"Score: {score:6.2f} | Max: {max_score:6.2f} | "
                    f"Loss: {agent._last_loss:.4f} | Learns: {agent.learns} | "
                    f"Speed: {ep_speed:.3f} ep/s"
                )

            # Save checkpoint periodically
            if episode % CHECKPOINT_INTERVAL == 0 and episode > 0:
                checkpoint_path = f"checkpoints/checkpoint_episode_{episode}.pth"
                agent.save_weights(checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")

            # Plot and save statistics periodically
            if episode % SAVE_INTERVAL == 0 and episode > 0:
                avg_score = sum(scores) / len(scores)
                last_100_avg.append(avg_score)

                # Create x-axis that matches the data points we have
                x_episodes = np.arange(
                    SAVE_INTERVAL, len(last_100_avg) * SAVE_INTERVAL + 1, SAVE_INTERVAL
                )

                plt.figure(figsize=(10, 6))
                plt.plot(x_episodes, last_100_avg)
                plt.xlabel("Episode")
                plt.ylabel("Average Score (last 100)")
                plt.title("Double DQN Training Progress - Pong (Self-Play)")
                plt.grid(True)
                plt.savefig("training_progress.png")
                plt.close()

                elapsed = time.time() - total_start_time
                eps_per_sec = episode / elapsed if elapsed > 0 else 0
                print(
                    f"📊 Avg score (last 100): {avg_score:.2f} | "
                    f"Episodes/sec: {eps_per_sec:.2f} | "
                    f"Memory: {len(agent.memory)}"
                )

    except KeyboardInterrupt:
        print("\n\n⏹️  Training interrupted by user")

    finally:
        # Save final weights
        agent.save_weights("final_weights.pth")
        print("\n✅ Final weights saved to final_weights.pth")
        agent.close()  # Close TensorBoard writer
        env_ai.close()
        env_selfplay.close()

        total_time = time.time() - total_start_time
        print(f"Training completed! Total time: {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()
