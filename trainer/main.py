"""
Main training script for Double DQN agent on Pong environment.
Features: 6-Phase Progressive Mastery Curriculum (PMC), TensorBoard logging, optimized hyperparameters.
Supports automatic checkpoint resuming to continue training from last saved state.

PMC Phases:
1. Ball Tracking (static opponent)
2. Basic Returns (slow AI)
3. First Wins (beginner AI) - Win-rate focused
4. Competitive Play (normal AI)
5. Advanced Control (reactive AI)
6. Mastery (self-play)
"""

import numpy as np
from collections import deque
import time
import re
import matplotlib.pyplot as plt
from trainer import environment
from trainer import Agent
from trainer.curriculum import CurriculumManager, create_env_for_phase
import sys
import os
from pathlib import Path

# Add project root to path so imports work when running directly
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """
    Find the latest checkpoint file based on episode number.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        tuple: (checkpoint_path, episode_number) or (None, 0) if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
    checkpoint_files = []
    pattern = re.compile(r"checkpoint_episode_(\d+)\.pth$")
    
    for filename in os.listdir(checkpoint_dir):
        match = pattern.match(filename)
        if match:
            episode_num = int(match.group(1))
            checkpoint_files.append((filename, episode_num))
    
    if not checkpoint_files:
        return None, 0
    
    # Sort by episode number and get the latest
    checkpoint_files.sort(key=lambda x: x[1], reverse=True)
    latest_file, latest_episode = checkpoint_files[0]
    
    return os.path.join(checkpoint_dir, latest_file), latest_episode


# Environment configuration
ENV_NAME = "PongHeadless-v0"  # Headless Pong environment for faster training
RENDER_MODE = None  # Set to 'human' for visualization, None for faster training

# Agent hyperparameters (OPTIMIZED v2 - improved learning signal)
POSSIBLE_ACTIONS = [0, 1, 2]  # 0: stay, 1: up, 2: down
STARTING_MEM_LEN = 10000  # Increased for more diverse initial experiences
MAX_MEM_LEN = 100000  # More recent experiences
LEARN_RATE = 0.001  # Doubled for faster learning
OBSERVATION_DIM = 9  # Pong observation space dimension
BATCH_SIZE = 128  # Doubled for more stable gradients

# Training configuration
MAX_EPISODES = 100000
SAVE_INTERVAL = 50  # Save plot every N episodes
CHECKPOINT_INTERVAL = 100  # Save model checkpoint every N episodes
TENSORBOARD_LOG = "./tensorboard_dqn/"  # TensorBoard log directory


def main():
    """Main training loop with 6-Phase Progressive Mastery Curriculum."""
    print("🏓 Double DQN Training with 6-Phase PMC (Progressive Mastery Curriculum)")
    print("=" * 60)

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

    # Initialize curriculum manager with larger window to reduce metric volatility
    curriculum = CurriculumManager(metrics_window=150)

    # Training statistics (will be restored from checkpoint if resuming)
    last_100_avg = []
    scores = deque(maxlen=100)
    max_score = float("-inf")
    total_start_time = time.time()
    previous_training_time = 0.0
    start_episode = 0

    # Automatic checkpoint detection and loading
    checkpoint_path, checkpoint_episode = find_latest_checkpoint("checkpoints")
    
    if checkpoint_path is not None:
        print(f"📂 Found checkpoint: {checkpoint_path}")
        training_state = agent.load_weights(checkpoint_path)
        start_episode = checkpoint_episode + 1  # Resume from next episode
        
        # Restore training state if available
        if training_state is not None:
            max_score = training_state.get("max_score", float("-inf"))
            previous_training_time = training_state.get("training_time_elapsed", 0.0)
            
            # Restore curriculum state
            if "curriculum_state" in training_state:
                curriculum.load_state(training_state["curriculum_state"])
            
            # Restore scores history
            saved_scores = training_state.get("scores_history", [])
            for s in saved_scores:
                scores.append(s)
            
            # Restore plotting history
            last_100_avg = training_state.get("last_100_avg", [])
            
            print(f"✅ Resumed training from episode {start_episode}")
            print(f"   Curriculum: {curriculum.get_status_string()}")
            print(f"   Max score: {max_score:.2f}")
            print(f"   Previous training time: {previous_training_time/60:.1f} min")
            print(f"   Restored {len(scores)} recent scores")
        else:
            # Legacy checkpoint without training state
            print(f"✅ Loaded weights from episode {checkpoint_episode}")
            print("   (Legacy checkpoint - training state not available)")
    else:
        print("📂 No checkpoint found, starting fresh training")

    if start_episode > 0:
        print(f"\n🔄 Resuming training from episode {start_episode} to {MAX_EPISODES}...")
    else:
        print(f"\n🆕 Starting fresh training for {MAX_EPISODES} episodes...")
    
    # Print curriculum phases
    print("\n📚 6-Phase Progressive Mastery Curriculum:")
    for i, phase in enumerate(curriculum.PHASES):
        marker = "→" if i == curriculum.current_phase else " "
        print(f"  {marker} Phase {i+1}: {phase.name} ({phase.opponent_type}, {phase.ball_speed}x speed)")
        if phase.advance_metric:
            print(f"       Advance: {phase.advance_metric} >= {phase.advance_threshold}")
            print(f"       Min: {phase.min_episodes} ep, Stability: {phase.stability_episodes} consecutive")
    
    print(f"\nRender mode: {RENDER_MODE}")
    print(f"Learning rate: {LEARN_RATE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Starting memory: {STARTING_MEM_LEN}")
    print(f"Checkpoints saved every {CHECKPOINT_INTERVAL} episodes")
    print(f"TensorBoard: tensorboard --logdir {TENSORBOARD_LOG}")
    print("-" * 60)

    # Create initial environment based on current phase
    phase_config = curriculum.get_phase_config()
    env = create_env_for_phase(phase_config, agent, RENDER_MODE)
    current_phase_index = curriculum.current_phase

    try:
        for episode in range(start_episode, MAX_EPISODES):
            timesteps_before = agent.total_timesteps
            episode_start_time = time.time()

            # Check if we need to recreate environment for new phase
            if curriculum.current_phase != current_phase_index:
                env.close()
                phase_config = curriculum.get_phase_config()
                env = create_env_for_phase(phase_config, agent, RENDER_MODE)
                current_phase_index = curriculum.current_phase
                print(f"\n🎯 Phase {current_phase_index + 1}: {curriculum.phase.name}")
                print(f"   Opponent: {phase_config['opponent_type']}, Ball speed: {phase_config['ball_speed_multiplier']}x")
                if phase_config['advance_metric']:
                    print(f"   Target: {phase_config['advance_metric']} >= {phase_config['advance_threshold']}")
                print()

            # Play one episode
            score, info = environment.play_episode_with_info(env, agent, debug=False)

            # Record metrics and check for phase advancement
            advanced = curriculum.record_episode(info)
            if advanced:
                print(f"\n🏆 PHASE COMPLETE! Advanced to Phase {curriculum.current_phase + 1}: {curriculum.phase.name}")
                print(f"   Metrics at completion: {curriculum.phase_history[-1]['final_metrics']}")
                print()

            # Update statistics
            episode_length = agent.total_timesteps - timesteps_before
            scores.append(score)
            if score > max_score:
                max_score = score

            # Log episode to TensorBoard
            agent.log_episode(episode, score, episode_length)
            
            # Log curriculum metrics to TensorBoard
            if agent.writer:
                metrics = curriculum.aggregate_metrics()
                agent.writer.add_scalar('Curriculum/Phase', curriculum.current_phase + 1, episode)
                agent.writer.add_scalar('Curriculum/Alignment', metrics['alignment'], episode)
                agent.writer.add_scalar('Curriculum/HitRate', metrics['hit_rate'], episode)
                agent.writer.add_scalar('Curriculum/AvgRally', metrics['rally'], episode)
                agent.writer.add_scalar('Curriculum/WinRate', metrics['win_rate'], episode)

            # Print episode statistics (less verbose for faster training)
            episode_duration = time.time() - episode_start_time

            if episode % 10 == 0:  # Print every 10 episodes
                ep_speed = 1.0 / episode_duration if episode_duration > 0 else 0
                phase_name = curriculum.phase.name[:12]  # Truncate for display
                print(
                    f"Ep {episode:5d} | P{curriculum.current_phase+1}:{phase_name:<12} | "
                    f"Steps: {episode_length:4d} | Score: {score:6.2f} | Max: {max_score:6.2f} | "
                    f"Loss: {agent._last_loss:.4f} | Speed: {ep_speed:.3f} ep/s"
                )

            # Save checkpoint periodically with full training state
            if episode % CHECKPOINT_INTERVAL == 0 and episode > 0:
                elapsed = time.time() - total_start_time + previous_training_time
                training_state = {
                    "episode": episode,
                    "curriculum_state": curriculum.get_state(),
                    "scores_history": list(scores),
                    "max_score": max_score,
                    "last_100_avg": last_100_avg.copy(),
                    "training_time_elapsed": elapsed,
                }
                checkpoint_path = f"checkpoints/checkpoint_episode_{episode}.pth"
                agent.save_weights(checkpoint_path, training_state=training_state)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
                print(f"   {curriculum.get_status_string()}")

            # Plot and save statistics periodically
            if episode % SAVE_INTERVAL == 0 and episode > 0:
                avg_score = sum(scores) / len(scores)
                last_100_avg.append(avg_score)

                # Create x-axis: each entry corresponds to SAVE_INTERVAL episodes
                x_episodes = np.arange(
                    SAVE_INTERVAL, len(last_100_avg) * SAVE_INTERVAL + 1, SAVE_INTERVAL
                )

                plt.figure(figsize=(12, 6))
                
                # Plot scores
                plt.subplot(1, 2, 1)
                plt.plot(x_episodes, last_100_avg)
                plt.xlabel("Episode")
                plt.ylabel("Average Score (last 100)")
                plt.title("Training Progress - Score")
                plt.grid(True)
                
                # Plot curriculum metrics
                plt.subplot(1, 2, 2)
                metrics = curriculum.aggregate_metrics()
                metric_names = list(metrics.keys())
                metric_values = list(metrics.values())
                plt.barh(metric_names, metric_values)
                plt.xlabel("Value")
                plt.title(f"Phase {curriculum.current_phase + 1}: {curriculum.phase.name}")
                plt.xlim(0, 1.2)
                
                plt.tight_layout()
                plt.savefig("training_progress.png")
                plt.close()

                session_time = time.time() - total_start_time
                total_episodes_run = episode - start_episode + 1
                eps_per_sec = total_episodes_run / session_time if session_time > 0 else 0
                print(
                    f"📊 Avg score (last 100): {avg_score:.2f} | "
                    f"Episodes/sec: {eps_per_sec:.2f} | "
                    f"Memory: {len(agent.memory)}"
                )
                print(f"   {curriculum.get_status_string()}")

    except KeyboardInterrupt:
        print("\n\n⏹️  Training interrupted by user")

    finally:
        # Save final weights with training state
        elapsed = time.time() - total_start_time + previous_training_time
        final_training_state = {
            "episode": episode if 'episode' in dir() else start_episode,
            "curriculum_state": curriculum.get_state(),
            "scores_history": list(scores),
            "max_score": max_score,
            "last_100_avg": last_100_avg.copy(),
            "training_time_elapsed": elapsed,
        }
        agent.save_weights("final_weights.pth", training_state=final_training_state)
        print("\n✅ Final weights saved to final_weights.pth")
        agent.close()  # Close TensorBoard writer
        env.close()

        session_time = time.time() - total_start_time
        total_time = session_time + previous_training_time
        print(f"Session time: {session_time/60:.1f} minutes")
        print(f"Total training time: {total_time/60:.1f} minutes")
        print(f"Final curriculum state: {curriculum.get_status_string()}")


if __name__ == "__main__":
    main()
