"""
Tests for PPO-optimized Pong environment and wrappers.
"""

import pytest
import numpy as np
import gymnasium as gym

from pong.env.pong_headless import PongHeadlessEnv, OpponentType
from pong.env.wrappers import WinFocusedRewardWrapper, EpisodeStatsWrapper, make_ppo_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from scripts.train_ppo_curriculum import create_eval_vec_env, evaluate_model


class TestPongHeadlessEnv:
    """Test the headless Pong environment."""

    def test_initialization(self):
        """Test environment initialization."""
        env = PongHeadlessEnv()
        assert env.observation_space.shape == (9,)
        assert env.action_space.n == 3
        env.close()

    def test_reset(self):
        """Test environment reset."""
        env = PongHeadlessEnv()
        obs, info = env.reset()

        assert obs.shape == (9,)
        assert "player_score" in info
        assert "opponent_score" in info
        assert info["player_score"] == 0
        assert info["opponent_score"] == 0

        env.close()

    def test_step(self):
        """Test environment step."""
        env = PongHeadlessEnv()
        obs, _ = env.reset()

        for action in [0, 1, 2]:  # Test all actions
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (9,)
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)

        env.close()

    def test_opponent_types(self):
        """Test different opponent types."""
        for opponent in OpponentType:
            env = PongHeadlessEnv(opponent_type=opponent)
            obs, _ = env.reset()
            obs, reward, terminated, truncated, info = env.step(0)
            assert obs.shape == (9,)
            env.close()

    def test_scoring(self):
        """Test that scoring works correctly."""
        env = PongHeadlessEnv(opponent_type=OpponentType.SLOW_AI)
        obs, _ = env.reset()

        # Play until someone scores
        for _ in range(10000):
            obs, reward, terminated, truncated, info = env.step(1)
            if info["player_score"] > 0 or info["opponent_score"] > 0:
                assert reward != 0  # Should get reward when scoring
                break

        env.close()

    def test_ball_acceleration(self):
        """Test that ball speed increases after paddle hits."""
        env = PongHeadlessEnv(opponent_type=OpponentType.SLOW_AI)
        obs, _ = env.reset()

        initial_speed = np.sqrt(env.ball_vx**2 + env.ball_vy**2)
        max_speed_seen = initial_speed
        hits_detected = 0

        for _ in range(5000):
            obs, reward, terminated, truncated, info = env.step(1)
            current_speed = np.sqrt(env.ball_vx**2 + env.ball_vy**2)
            if current_speed > max_speed_seen + 0.01:
                max_speed_seen = current_speed
                hits_detected += 1
            if terminated:
                obs, _ = env.reset()
                break

        # Ball should have accelerated during the episode
        assert (
            max_speed_seen > initial_speed
        ), f"Ball should accelerate: initial={initial_speed:.2f}, max={max_speed_seen:.2f}"
        env.close()

    def test_ball_speed_capped(self):
        """Test that horizontal ball speed is capped at max_speed_mult * launch speed."""
        from pong.env.pong_headless import GameConfig

        config = GameConfig(ball_accel_per_hit=0.50, ball_max_speed_mult=1.5)
        env = PongHeadlessEnv(opponent_type=OpponentType.SLOW_AI, config=config)
        obs, _ = env.reset()

        launch_speed = config.base_ball_speed * env.ball_speed_multiplier
        max_allowed = launch_speed * config.ball_max_speed_mult

        for _ in range(10000):
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            hspeed = abs(env.ball_vx)
            assert (
                hspeed <= max_allowed + 0.5
            ), f"Horizontal speed {hspeed:.2f} exceeds cap {max_allowed:.2f}"
            if terminated:
                obs, _ = env.reset()

        env.close()

    def test_angle_from_offset_model(self):
        """Test that bounce angle is determined by paddle hit position."""
        env = PongHeadlessEnv(opponent_type=OpponentType.SLOW_AI)

        # Manually position ball and paddle to force a center hit
        env.reset()
        c = env.config

        # Set ball moving right toward player paddle, centered on paddle
        env.ball_x = env.player_x - c.paddle_width - 1
        env.ball_y = env.player_y  # Center hit
        speed = c.base_ball_speed * env.ball_speed_multiplier
        env.ball_vx = speed
        env.ball_vy = 0.0

        # Step to trigger collision
        env.step(0)

        # Center hit should produce near-zero vertical velocity
        assert (
            abs(env.ball_vy) < 0.5
        ), f"Center hit should have near-zero vy, got {env.ball_vy:.2f}"
        assert (
            env.ball_vx < 0
        ), "Ball should bounce leftward after hitting player paddle"

        env.close()

    def test_edge_hit_sharper_than_center(self):
        """Test that edge hits produce much steeper deflection than mid-paddle hits (Atari Pong style)."""
        from pong.env.pong_headless import GameConfig

        config = GameConfig()
        env = PongHeadlessEnv(opponent_type=OpponentType.SLOW_AI, config=config)
        env.reset()
        c = env.config
        speed = c.base_ball_speed * env.ball_speed_multiplier

        # Mid-paddle hit: offset=0.5
        env.ball_x = env.player_x - c.paddle_width - 1
        env.ball_y = env.player_y + c.paddle_height * 0.25  # offset ~0.5
        env.ball_vx = speed
        env.ball_vy = 0.0
        env.step(0)
        mid_vy = abs(env.ball_vy)

        # Edge hit: offset=1.0
        env.ball_x = env.player_x - c.paddle_width - 1
        env.ball_y = env.player_y + c.paddle_height * 0.5  # offset ~1.0
        env.ball_vx = speed
        env.ball_vy = 0.0
        env.step(0)
        edge_vy = abs(env.ball_vy)

        # Edge should deflect significantly more than mid
        assert (
            edge_vy > mid_vy * 1.5
        ), f"Edge hit vy ({edge_vy:.2f}) should be >1.5x mid hit vy ({mid_vy:.2f})"

        env.close()


class TestWinFocusedRewardWrapper:
    """Test the win-focused reward wrapper."""

    def test_wrapper_initialization(self):
        """Test wrapper can be created."""
        env = PongHeadlessEnv()
        wrapped = WinFocusedRewardWrapper(env)
        assert wrapped.observation_space.shape == (9,)
        wrapped.close()

    def test_reward_structure(self):
        """Test that rewards are modified correctly.

        With hit_reward=0 by default, rewards come from:
        - Scoring (+5 point_reward)
        - Pressure shaping (+0.02 max)
        - Tracking (+0.01 max)
        - Step penalty (-0.001)
        """
        env = PongHeadlessEnv(opponent_type=OpponentType.SLOW_AI)
        wrapped = WinFocusedRewardWrapper(
            env,
            point_reward=5.0,
            win_bonus=10.0,
            hit_reward=0.0,  # Disabled
            pressure_scale=0.02,
            step_penalty=0.001,
        )

        obs, _ = wrapped.reset()

        # Play until someone scores
        for _ in range(10000):
            obs, reward, terminated, truncated, info = wrapped.step(1)

            # Check that point rewards are applied on scoring
            if info["player_score"] > 0:
                # point_reward (5.0) minus step penalty and pressure/tracking adjustments
                assert reward >= 4.5
                break
            elif info["opponent_score"] > 0:
                # -point_reward (-5.0) plus possible small dense rewards
                assert reward <= -4.5
                break

        wrapped.close()

    def test_game_win_bonus(self):
        """Test that game win bonus is applied."""
        env = PongHeadlessEnv(opponent_type=OpponentType.SLOW_AI, max_score=1)
        wrapped = WinFocusedRewardWrapper(
            env,
            point_reward=5.0,
            win_bonus=10.0,
            hit_reward=0.0,  # Disabled
        )

        obs, _ = wrapped.reset()
        total_reward = 0.0

        # Play until game ends
        for _ in range(10000):
            obs, reward, terminated, truncated, info = wrapped.step(
                env.action_space.sample()
            )
            total_reward += reward
            if terminated:
                # Winning: point_reward (5) + win_bonus (10) = 15 + small dense rewards
                # Losing: -point_reward (-5) + -win_bonus (-10) = -15 + small dense rewards
                if info["player_score"] > info["opponent_score"]:
                    assert total_reward > 10.0  # Won with bonus
                else:
                    assert total_reward < -5.0  # Lost with penalty
                break

        wrapped.close()

    def test_pressure_reward_bounded(self):
        """Test that pressure reward is positive and bounded when ball moves toward opponent."""
        env = PongHeadlessEnv(opponent_type=OpponentType.SLOW_AI)
        wrapped = WinFocusedRewardWrapper(
            env,
            hit_reward=0.0,
            pressure_scale=0.05,  # Default pressure scale
            step_penalty=0.0,  # Disable step penalty for cleaner test
            tracking_reward=0.0,  # Disable tracking for cleaner test
            point_reward=0.0,  # Disable scoring rewards for cleaner test
            win_bonus=0.0,
        )

        # Seed the launch sequence and track the ball so the paddle actually
        # returns it. Pressure is paid only on a return, so a policy that never
        # reaches the ball would leave nothing to assert on.
        obs, _ = wrapped.reset(seed=12345)
        pressure_rewards = []

        for _ in range(2000):
            # 1 moves the paddle up (towards smaller y), 2 moves it down.
            action = 1 if obs[1] < obs[4] else 2
            obs, reward, terminated, truncated, info = wrapped.step(action)

            ball_vx = obs[2]  # Ball x velocity

            if ball_vx < 0:  # Ball moving toward opponent
                pressure_rewards.append(reward)

            if terminated:
                obs, _ = wrapped.reset()

        # Verify we collected enough samples
        assert (
            len(pressure_rewards) > 10
        ), f"Should have pressure situations, got {len(pressure_rewards)}"

        # Pressure rewards should be non-negative and bounded
        for r in pressure_rewards:
            assert r >= 0.0, f"Pressure reward should be >= 0, got {r}"
            assert (
                r <= 0.05
            ), f"Pressure reward should be <= 0.05 (pressure_scale), got {r}"

        # Some pressure rewards should be positive (when ball is heading away from opponent)
        assert (
            max(pressure_rewards) > 0.0
        ), "At least some pressure rewards should be positive"

        # Pressure is paid ONCE per return, not on every step the ball travels
        # toward the opponent, so paid steps must be a small minority. This is
        # the regression guard against reverting to per-step payment.
        paid = [r for r in pressure_rewards if r > 0.0]
        assert len(paid) * 10 < len(pressure_rewards), (
            f"Pressure should be paid once per return, not per step: "
            f"{len(paid)} paid steps out of {len(pressure_rewards)}"
        )

        wrapped.close()

    def test_step_penalty_applied(self):
        """Test that step penalty is applied each step."""
        env = PongHeadlessEnv(opponent_type=OpponentType.SLOW_AI)
        wrapped = WinFocusedRewardWrapper(
            env,
            hit_reward=0.0,
            pressure_scale=0.0,
            step_penalty=0.01,
            tracking_reward=0.0,
            point_reward=0.0,
            win_bonus=0.0,
        )

        obs, _ = wrapped.reset()
        total_reward = 0.0
        steps = 0

        for _ in range(100):
            obs, reward, terminated, truncated, info = wrapped.step(0)  # Stay still
            total_reward += reward
            steps += 1
            if terminated:
                break

        # With only step_penalty=0.01, total reward should be approximately -steps * 0.01
        expected_reward = -steps * 0.01
        assert (
            abs(total_reward - expected_reward) < 0.1
        ), f"Expected ~{expected_reward:.2f}, got {total_reward:.2f}"

        wrapped.close()


class TestEpisodeStatsWrapper:
    """Test the episode stats wrapper."""

    def test_stats_tracking(self):
        """Test that episode stats are tracked."""
        env = PongHeadlessEnv()
        wrapped = EpisodeStatsWrapper(env)

        obs, info = wrapped.reset()
        assert "episode_stats" not in info  # Not present at reset

        # Play until episode ends
        for _ in range(10000):
            obs, reward, terminated, truncated, info = wrapped.step(1)
            if terminated or truncated:
                assert "episode_stats" in info
                assert "reward" in info["episode_stats"]
                assert "length" in info["episode_stats"]
                assert "won" in info["episode_stats"]
                break

        wrapped.close()


class TestMakePPOEnv:
    """Test the make_ppo_env helper function."""

    def test_creates_wrapped_env(self):
        """Test that make_ppo_env creates properly wrapped environment."""
        env = make_ppo_env("slow_ai")

        # Should be wrapped
        assert isinstance(env, EpisodeStatsWrapper)

        # Test it works
        obs, _ = env.reset()
        assert obs.shape == (9,)

        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (9,)

        env.close()

    def test_different_opponents(self):
        """Test creating environments with different opponents."""
        for opponent in ["slow_ai", "beginner_ai", "normal_ai", "reactive_ai"]:
            env = make_ppo_env(opponent)
            obs, _ = env.reset()
            assert obs.shape == (9,)
            env.close()

    def test_default_reward_params(self):
        """Test that make_ppo_env creates env with correct default reward params."""
        env = make_ppo_env("slow_ai")

        # Access the wrapped WinFocusedRewardWrapper
        reward_wrapper = env.env  # EpisodeStatsWrapper wraps WinFocusedRewardWrapper
        assert (
            reward_wrapper.hit_reward == 0.0
        ), "hit_reward should be 0.0 by default (disabled)"
        assert (
            reward_wrapper.pressure_scale == 0.02
        ), "pressure_scale should be 0.02 by default"
        assert (
            reward_wrapper.step_penalty == 0.001
        ), "step_penalty should be 0.001 by default"

        env.close()


def test_eval_env_matches_vec_type():
    """Eval env should be a VecEnv (Subproc or Dummy) matching requested type."""
    eval_env_sub = create_eval_vec_env(
        OpponentType.SLOW_AI, ball_speed=1.0, use_subproc=True
    )
    assert isinstance(eval_env_sub, SubprocVecEnv)
    eval_env_sub.close()

    eval_env_dummy = create_eval_vec_env(
        OpponentType.SLOW_AI, ball_speed=1.0, use_subproc=False
    )
    assert isinstance(eval_env_dummy, DummyVecEnv)
    eval_env_dummy.close()


def test_environment_integration():
    """Integration test: full episode with all wrappers."""
    env = make_ppo_env(
        opponent_type="slow_ai",
        ball_speed=1.0,
        max_score=5,
        max_steps=5000,
    )

    obs, info = env.reset()
    total_reward = 0.0
    steps = 0

    # Play full episode
    while steps < 10000:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            # Verify info dict
            assert "episode_stats" in info
            assert info["episode_stats"]["length"] == steps
            assert abs(info["episode_stats"]["reward"] - total_reward) < 0.1
            break

    env.close()
    print(f"✓ Integration test passed: {steps} steps, {total_reward:.2f} reward")


def test_evaluate_model_accepts_phase_params():
    """Smoke test: evaluate_model should accept ball_speed/max_steps args."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    def _make_env():
        env = PongHeadlessEnv(
            opponent_type=OpponentType.SLOW_AI,
            ball_speed_multiplier=0.5,
            max_score=5,
            max_steps=5000,
        )
        return Monitor(env)

    model = PPO("MlpPolicy", DummyVecEnv([_make_env]), verbose=0)

    results = evaluate_model(
        model,
        OpponentType.SLOW_AI,
        ball_speed=0.5,
        max_steps=5000,
        n_episodes=1,
        deterministic=True,
        verbose=False,
    )

    assert results["opponent"] == "slow_ai"
    assert results["ball_speed"] == 0.5
    assert results["max_steps"] == 5000
    assert results["n_episodes"] == 1
    assert 0.0 <= results["win_rate"] <= 1.0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
