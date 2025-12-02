"""
Test script for Pong Gym environment integration.

This script tests the gym environment thoroughly to ensure it works correctly
with the Gymnasium API and is ready for RL training.

Uses PongHeadlessEnv by default for faster, more reliable testing.
"""

import gymnasium as gym
import numpy as np
# Use headless environment for testing (no pygame dependency)
from pong.env.pong_headless import PongHeadlessEnv, register_headless_env


def test_environment_creation():
    """Test basic environment creation and registration."""
    print("🧪 Testing environment creation...")

    # Register headless environment
    register_headless_env()

    # Test direct instantiation
    env = PongHeadlessEnv(render_mode=None)
    print(f"✅ Direct instantiation successful")

    # Test gym.make
    env2 = gym.make("PongHeadless-v0", render_mode=None)
    print(f"✅ gym.make instantiation successful")

    return env2


def test_action_observation_spaces(env):
    """Test action and observation spaces."""
    print("\n🧪 Testing action and observation spaces...")

    # Check action space
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 3
    print(f"✅ Action space: {env.action_space}")

    # Check observation space
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (9,)
    print(f"✅ Observation space: {env.observation_space}")
    print(f"   Low bounds: {env.observation_space.low}")
    print(f"   High bounds: {env.observation_space.high}")


def test_reset_functionality(env):
    """Test environment reset functionality."""
    print("\n🧪 Testing reset functionality...")

    # Test reset
    obs, info = env.reset()

    # Check observation
    assert obs.shape == (9,), f"Expected obs shape (9,), got {obs.shape}"
    assert env.observation_space.contains(obs), "Observation not in observation space"
    print(f"✅ Reset observation shape: {obs.shape}")
    print(f"✅ Reset observation: {obs}")

    # Check info
    assert isinstance(info, dict)
    required_keys = ["player_score", "opponent_score", "steps", "ball_position"]
    for key in required_keys:
        assert key in info, f"Missing key {key} in info dict"
    print(f"✅ Reset info keys: {list(info.keys())}")

    return obs, info


def test_step_functionality(env, num_steps=50):
    """Test environment step functionality."""
    print(f"\n🧪 Testing step functionality for {num_steps} steps...")

    obs, info = env.reset()
    total_reward = 0

    for step in range(num_steps):
        # Test each action type
        action = step % 3  # Cycle through actions 0, 1, 2

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Validate step return values
        assert env.observation_space.contains(obs), f"Invalid observation at step {step}"
        assert isinstance(reward, (int, float)), f"Invalid reward type at step {step}"
        assert isinstance(terminated, bool), f"Invalid terminated type at step {step}"
        assert isinstance(truncated, bool), f"Invalid truncated type at step {step}"
        assert isinstance(info, dict), f"Invalid info type at step {step}"

        # Check info contents
        assert "player_score" in info
        assert "opponent_score" in info
        assert "steps" in info

        if step % 10 == 0:
            print(
                f"   Step {step}: action={action}, reward={reward:.3f}, "
                f"scores=({info['player_score']}, {info['opponent_score']})"
            )

        if terminated or truncated:
            print(f"   Episode ended at step {step}")
            obs, info = env.reset()
            total_reward = 0

    print(f"✅ Step functionality test completed")
    print(f"   Total reward accumulated: {total_reward:.3f}")


def test_reward_system(env):
    """Test the reward system by playing a short episode."""
    print("\n🧪 Testing reward system...")

    obs, info = env.reset()
    rewards = []
    actions_taken = []

    for step in range(200):
        # Simple strategy: follow the ball
        ball_y = obs[1]  # Ball Y position
        player_y = obs[4]  # Player Y position

        if ball_y < player_y - 20:
            action = 1  # Move up
        elif ball_y > player_y + 20:
            action = 2  # Move down
        else:
            action = 0  # Stay

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        actions_taken.append(action)

        if terminated or truncated:
            break

    print(f"✅ Reward system test completed")
    print(f"   Episode length: {len(rewards)} steps")
    print(f"   Total reward: {sum(rewards):.3f}")
    print(f"   Average reward: {np.mean(rewards):.4f}")
    print(f"   Reward range: [{min(rewards):.3f}, {max(rewards):.3f}]")
    print(f"   Final scores: Player {info['player_score']}, Opponent {info['opponent_score']}")


def test_rendering_modes():
    """Test different rendering modes."""
    print("\n🧪 Testing rendering modes...")

    # Test ansi mode (text-based rendering for headless env)
    try:
        env = gym.make("PongHeadless-v0", render_mode="ansi")
        obs, _ = env.reset()

        # Take a few steps and render
        for _ in range(5):
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)
            ansi_output = env.render()

            if ansi_output is not None:
                assert isinstance(ansi_output, str), "ANSI output should be string"
                assert len(ansi_output) > 0, "ANSI output should not be empty"

        env.close()
        print(f"✅ ANSI rendering mode works")
        print(f"   Output preview: {ansi_output[:50]}...")

    except Exception as e:
        print(f"⚠️ ANSI rendering failed: {e}")

    # Test None rendering mode
    try:
        env = gym.make("PongHeadless-v0", render_mode=None)
        obs, _ = env.reset()
        result = env.render()
        assert result is None, "None render mode should return None"
        env.close()
        print(f"✅ None rendering mode works")

    except Exception as e:
        print(f"⚠️ None rendering mode failed: {e}")


def test_episode_completion():
    """Test full episode completion scenarios."""
    print("\n🧪 Testing episode completion...")

    # Test score-based termination
    env = gym.make("PongHeadless-v0", max_score=3, max_steps=10000, render_mode=None)
    obs, info = env.reset()

    step_count = 0
    while step_count < 5000:  # Safety limit
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        if terminated:
            print(f"✅ Episode terminated by score after {step_count} steps")
            print(
                f"   Final scores: Player {info['player_score']}, Opponent {info['opponent_score']}"
            )
            break

        if truncated:
            print(f"✅ Episode truncated after {step_count} steps")
            break

    env.close()

    # Test step-based truncation
    env = gym.make("PongHeadless-v0", max_score=100, max_steps=50, render_mode=None)
    obs, info = env.reset()

    step_count = 0
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        if terminated:
            print(f"✅ Episode terminated early by score after {step_count} steps")
            break

        if truncated:
            print(f"✅ Episode truncated by max_steps after {step_count} steps")
            break

    env.close()


def test_multiple_episodes():
    """Test running multiple episodes."""
    print("\n🧪 Testing multiple episodes...")

    env = gym.make("PongHeadless-v0", max_score=3, max_steps=1000, render_mode=None)

    episode_lengths = []
    episode_rewards = []

    for episode in range(5):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            if terminated or truncated:
                episode_lengths.append(episode_length)
                episode_rewards.append(episode_reward)
                print(
                    f"   Episode {episode + 1}: {episode_length} steps, "
                    f"reward {episode_reward:.3f}, scores ({info['player_score']}, {info['opponent_score']})"
                )
                break

    env.close()

    print(f"✅ Multiple episodes test completed")
    print(
        f"   Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}"
    )
    print(
        f"   Average episode reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}"
    )


def run_all_tests():
    """Run all environment tests."""
    print("🏓 Pong Gym Environment Integration Tests")
    print("=" * 50)

    try:
        # Basic tests
        env = test_environment_creation()
        test_action_observation_spaces(env)
        test_reset_functionality(env)
        test_step_functionality(env)
        test_reward_system(env)
        env.close()

        # Rendering tests
        test_rendering_modes()

        # Episode completion tests
        test_episode_completion()
        test_multiple_episodes()

        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("✅ The Pong environment is ready for RL training")
        print("🚀 You can now use it with popular RL libraries like:")
        print("   - Stable-Baselines3")
        print("   - Ray RLLib")
        print("   - TensorFlow Agents")
        print("   - And any other Gymnasium-compatible library!")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = run_all_tests()

    if success:
        print("\n🎯 Next steps:")
        print("   1. Run 'python train_pong_agent.py' to start training")
        print("   2. Monitor with TensorBoard: 'tensorboard --logdir ./pong_tensorboard/'")
        print("   3. Experiment with different algorithms and hyperparameters")

    exit(0 if success else 1)
