"""
Example training script for Pong RL agent using Stable-Baselines3.

This script demonstrates how to train a reinforcement learning agent to play Pong
using the PPO algorithm from Stable-Baselines3.
"""

import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from pong import register_pong_env


def create_training_environment(n_envs: int = 4, max_score: int = 11):
    """Create vectorized training environment."""

    def make_env():
        env = gym.make("Pong-v0", max_score=max_score, max_steps=5000)
        env = Monitor(env)
        return env

    return make_vec_env(make_env, n_envs=n_envs)


def train_ppo_agent(total_timesteps: int = 200000, save_path: str = "./models/"):
    """Train a PPO agent to play Pong."""
    print("🚀 Training PPO Agent for Pong...")

    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)

    # Create training environment
    train_env = create_training_environment(n_envs=4, max_score=5)  # Shorter games for training

    # Create evaluation environment
    eval_env = Monitor(gym.make("Pong-v0", max_score=5, max_steps=5000))

    # Create callbacks
    # Stop training when the model reaches the reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=4.0, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        best_model_save_path=save_path,
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=1,
    )

    # Create PPO model with optimized hyperparameters
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./pong_tensorboard/",
    )

    # Train the model
    print(f"🎯 Starting training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Save the final model
    model.save(os.path.join(save_path, "pong_ppo_final"))
    print(f"✅ Training completed! Model saved to {save_path}")

    return model, eval_env


def train_dqn_agent(total_timesteps: int = 100000, save_path: str = "./models/"):
    """Train a DQN agent to play Pong."""
    print("🚀 Training DQN Agent for Pong...")

    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)

    # Create environment
    train_env = Monitor(gym.make("Pong-v0", max_score=5, max_steps=5000))
    eval_env = Monitor(gym.make("Pong-v0", max_score=5, max_steps=5000))

    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path="./logs/",
        eval_freq=2000,
        deterministic=True,
        render=False,
        verbose=1,
    )

    # Create DQN model
    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        target_update_interval=1000,
        train_freq=4,
        gradient_steps=1,
        exploration_final_eps=0.05,
        exploration_fraction=0.1,
        verbose=1,
        tensorboard_log="./pong_tensorboard/",
    )

    # Train the model
    print(f"🎯 Starting training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Save the final model
    model.save(os.path.join(save_path, "pong_dqn_final"))
    print(f"✅ Training completed! Model saved to {save_path}")

    return model, eval_env


def test_agent(model_path: str, episodes: int = 5, render: bool = True):
    """Test a trained agent."""
    print(f"🎮 Testing agent from {model_path}...")

    # Load the model
    if "ppo" in model_path.lower():
        model = PPO.load(model_path)
    elif "dqn" in model_path.lower():
        model = DQN.load(model_path)
    else:
        # Try PPO first, then DQN
        try:
            model = PPO.load(model_path)
        except Exception:
            model = DQN.load(model_path)

    # Create test environment
    env = gym.make("Pong-v0", render_mode="human" if render else None, max_score=11)

    episode_rewards = []
    episode_lengths = []
    wins = 0

    for episode in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0

        print(f"\n📺 Episode {episode + 1}/{episodes}")

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            if render:
                env.render()

            if terminated or truncated:
                player_score = info["player_score"]
                opponent_score = info["opponent_score"]

                if player_score > opponent_score:
                    wins += 1
                    result = "🏆 WIN"
                else:
                    result = "❌ LOSS"

                print(f"{result} - Final Score: Player {player_score}, Opponent {opponent_score}")
                print(f"Episode Reward: {episode_reward:.2f}, Length: {episode_length} steps")
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    env.close()

    # Print summary
    print(f"\n📊 Test Results Summary:")
    print(f"Win Rate: {wins}/{episodes} ({100 * wins / episodes:.1f}%)")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")

    return episode_rewards, episode_lengths


def plot_training_results(log_dir: str = "./logs/"):
    """Plot training results from TensorBoard logs."""
    try:
        # Optional imports for plotting
        try:
            import pandas as pd
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:
            print("pandas and tensorboard required for plotting")
            return

        print("📈 Plotting training results...")

        # This is a simplified version - you might want to use TensorBoard directly
        # for more detailed analysis
        print("💡 For detailed training plots, run:")
        print("   tensorboard --logdir ./pong_tensorboard/")
        print("   Then open http://localhost:6006 in your browser")

    except ImportError:
        print("📈 Install tensorboard and pandas for plotting: pip install tensorboard pandas")


def main():
    """Main training script."""
    print("🏓 Pong RL Training Script")
    print("=" * 50)

    # Register the environment
    register_pong_env()

    # Choose algorithm
    algorithm = input("Choose algorithm (ppo/dqn) [ppo]: ").lower() or "ppo"

    # Training parameters
    if algorithm == "ppo":
        timesteps = int(input("Training timesteps [200000]: ") or 200000)
        model, env = train_ppo_agent(timesteps)
        model_path = "./models/pong_ppo_final"
    else:
        timesteps = int(input("Training timesteps [100000]: ") or 100000)
        model, env = train_dqn_agent(timesteps)
        model_path = "./models/pong_dqn_final"

    env.close()

    # Test the agent
    test_episodes = int(input("Test episodes [5]: ") or 5)
    test_agent(model_path, episodes=test_episodes, render=True)

    # Show plotting instructions
    plot_training_results()

    print("\n✅ Training and testing completed!")
    print("🎯 Next steps:")
    print("   - Monitor training with TensorBoard")
    print("   - Experiment with hyperparameters")
    print("   - Try different reward functions")
    print("   - Implement multi-agent training")


if __name__ == "__main__":
    main()
