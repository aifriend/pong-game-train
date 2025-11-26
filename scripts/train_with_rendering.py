"""
Training script for Pong with visual rendering enabled.
Uses Stable-Baselines3 (PyTorch-based) with rendering to watch the agent learn.
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from pong import register_pong_env


def make_env_with_rendering():
    """Create environment with rendering enabled."""

    def _init():
        env = gym.make("Pong-v0", render_mode="human", max_score=5, max_steps=5000)
        env = Monitor(env)
        return env

    return _init


def train_with_rendering(total_timesteps: int = 100000):
    """
    Train a PPO agent with rendering enabled to watch learning progress.

    Args:
        total_timesteps: Total training timesteps (default: 100000)
                        Note: Lower than normal because rendering slows training
    """
    print("🏓 Training Pong with Visual Rendering")
    print("=" * 60)
    print("⚠️  Note: Rendering will slow down training significantly")
    print("    You'll see the game window showing each training step")
    print("    Press Ctrl+C to stop training early")
    print("=" * 60)

    # Register environment
    register_pong_env()

    # Create directories
    os.makedirs("./models/", exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)

    # Create single environment with rendering (not vectorized for visual feedback)
    env = gym.make("Pong-v0", render_mode="human", max_score=5, max_steps=5000)
    env = Monitor(env, "./logs/")

    print("\n🎯 Environment: Pong-v0")
    print("   Render mode: human (visible)")
    print("   Max score: 5")
    print(f"   Training timesteps: {total_timesteps}")
    print("\n🧠 Creating PPO agent...")

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./pong_tensorboard/",
    )

    print("\n🚀 Starting training...")
    print("   Watch the game window to see the agent learn!")
    print("-" * 60)

    try:
        # Train the model
        model.learn(total_timesteps=total_timesteps, progress_bar=True)

        # Save the model
        model_path = "./models/pong_ppo_rendering"
        model.save(model_path)
        print("\n✅ Training completed!")
        print(f"   Model saved to: {model_path}.zip")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        model_path = "./models/pong_ppo_interrupted"
        model.save(model_path)
        print(f"   Model saved to: {model_path}.zip")

    finally:
        env.close()
        print("\n🎮 Training session ended")


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("  PONG REINFORCEMENT LEARNING - VISUAL TRAINING")
    print("=" * 60)

    # Ask user for training duration
    print("\nHow many timesteps would you like to train?")
    print("  • 10,000   - Quick test (a few minutes)")
    print("  • 50,000   - Short training")
    print("  • 100,000  - Medium training (recommended)")
    print("  • 200,000+ - Long training")

    timesteps_input = input("\nEnter timesteps [100000]: ").strip()

    if timesteps_input == "":
        timesteps = 100000
    else:
        try:
            timesteps = int(timesteps_input)
        except ValueError:
            print("Invalid input, using default: 100000")
            timesteps = 100000

    # Start training
    train_with_rendering(total_timesteps=timesteps)

    print("\n💡 Next steps:")
    print("   • View training metrics: tensorboard --logdir ./pong_tensorboard/")
    print("   • Test the trained agent: python scripts/play.py")
    print("   • Continue training: Load the saved model and call learn() again")


if __name__ == "__main__":
    main()
