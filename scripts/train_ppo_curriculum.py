#!/usr/bin/env python3
"""
PPO Training Script with 5-Phase Curriculum for Pong.

This script uses Stable-Baselines3 PPO with a graduated curriculum
to train an agent that focuses on WINNING games rather than rallying.

Curriculum Phases:
1. Easy Wins (100K)   - SLOW_AI (40% speed), 0.6x ball, high hit reward
2. Beginner (100K)    - BEGINNER_AI (55% speed), 0.7x ball, moderate hit reward
3. Medium (150K)      - MEDIUM_AI (65% speed), 0.8x ball, low hit reward
4. Competitive (200K) - NORMAL_AI (70% speed), 0.9x ball, no hit reward
5. Master (300K)      - REACTIVE_AI (90% speed + prediction), 1.0x ball

Key features:
- Gradual opponent speed ramp with small increments (~10% each)
- Graduated reward shaping: high hit+pressure → pure win-focus
- Pressure shaping paid once per return (0.15 decaying to 0.08), not per step
- Learning rate & entropy decay across phases
- Regression detection (stops if agent forgets basics)
- Parallel environments for faster training

Usage:
    python scripts/train_ppo_curriculum.py [--phase 1|2|3|4|5]
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from pong.env.pong_headless import PongHeadlessEnv, OpponentType
from pong.env.wrappers import WinFocusedRewardWrapper, EpisodeStatsWrapper

# 5-Phase Curriculum Configuration
# Small difficulty increments (~10% speed per phase) + graduated reward shaping
# pressure_scale is the WHOLE payment for one return, not a per-step rate; a
# perfectly placed return pays the full value, so 1.0 is worth 20% of the 5.0
# point reward. Retuned for one-shot payment: the inherited 0.15 -> 0.08 values
# dated from per-step accumulation and left the placement signal at ~3% of a
# point, effectively switched off.
# hit_reward should fade gradually so the agent always has learning signal.
CURRICULUM_PHASES = {
    1: {
        "name": "Easy Wins",
        "opponent_type": OpponentType.SLOW_AI,
        "ball_speed": 0.6,
        "timesteps": 100_000,
        "target_win_rate": 0.60,
        "description": "Learn basics: hitting ball + scoring against slow AI",
        # Strong hit reward to bootstrap motor skills + strong pressure for offense
        "hit_reward": 0.5,
        "tracking_reward": 0.02,
        "pressure_scale": 1.0,
        "step_penalty": 0.001,
        # Hyperparameters - high LR for fast initial learning
        "learning_rate": 3e-4,
        "ent_coef": 0.05,
    },
    2: {
        "name": "Beginner",
        "opponent_type": OpponentType.BEGINNER_AI,
        "ball_speed": 0.7,
        "timesteps": 100_000,
        "target_win_rate": 0.50,
        "description": "Transfer skills to slightly faster opponent",
        # Reduced hit reward, maintained pressure
        "hit_reward": 0.3,
        "tracking_reward": 0.02,
        "pressure_scale": 1.0,
        "step_penalty": 0.001,
        # Hyperparameters
        "learning_rate": 2.5e-4,
        "ent_coef": 0.04,
    },
    3: {
        "name": "Medium",
        "opponent_type": OpponentType.MEDIUM_AI,
        "ball_speed": 0.8,
        "timesteps": 150_000,
        "target_win_rate": 0.40,
        "description": "Learn offensive placement against competent opponent",
        # Low hit reward - transitioning to win-focused
        "hit_reward": 0.1,
        "tracking_reward": 0.015,
        "pressure_scale": 0.8,
        "step_penalty": 0.001,
        # Hyperparameters
        "learning_rate": 1.5e-4,
        "ent_coef": 0.03,
    },
    4: {
        "name": "Competitive",
        "opponent_type": OpponentType.NORMAL_AI,
        "ball_speed": 0.9,
        "timesteps": 200_000,
        "target_win_rate": 0.30,
        "description": "Beat a fast opponent with near-full ball speed",
        # Keep small hit reward to prevent value function collapse
        "hit_reward": 0.05,
        "tracking_reward": 0.01,
        "pressure_scale": 0.65,
        "step_penalty": 0.001,
        # Hyperparameters - VERY low LR to prevent catastrophic forgetting
        "learning_rate": 5e-5,
        "ent_coef": 0.015,
    },
    5: {
        "name": "Master",
        "opponent_type": OpponentType.REACTIVE_AI,
        "ball_speed": 1.0,
        "timesteps": 200_000,
        "target_win_rate": 0.20,
        "description": "Master the predictive AI at full speed",
        # Keep small hit reward + moderate pressure to prevent value collapse
        "hit_reward": 0.05,
        "tracking_reward": 0.01,
        "pressure_scale": 0.5,
        "step_penalty": 0.001,
        # Hyperparameters - VERY low LR to prevent catastrophic forgetting
        "learning_rate": 3e-5,
        "ent_coef": 0.01,
        # Mixed opponents: 4 REACTIVE + 2 NORMAL + 2 MEDIUM to prevent forgetting
        "mixed_opponents": [
            (OpponentType.REACTIVE_AI, 1.0, 4),
            (OpponentType.NORMAL_AI, 0.9, 2),
            (OpponentType.MEDIUM_AI, 0.8, 2),
        ],
    },
}

NUM_PHASES = len(CURRICULUM_PHASES)


class CurriculumCallback(BaseCallback):
    """
    Callback for curriculum phase transitions and logging.

    Tracks win rates with clean, single-line progress updates.
    """

    def __init__(
        self,
        phase: int,
        target_win_rate: float,
        log_freq: int = 10000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.phase = phase
        self.target_win_rate = target_win_rate
        self.log_freq = log_freq

        # Tracking
        self.wins = 0
        self.games = 0
        self.last_log_step = 0

    def _on_step(self) -> bool:
        """Called at each step."""
        # Check for episode completions in info
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode_stats" in info:
                    stats = info["episode_stats"]
                    if stats["won"]:
                        self.wins += 1
                    self.games += 1

        # Periodic progress update (single line)
        if self.num_timesteps - self.last_log_step >= self.log_freq:
            self._log_progress()
            self.last_log_step = self.num_timesteps

        return True

    def _log_progress(self):
        """Log training progress as single line."""
        if self.games == 0:
            return

        win_rate = self.wins / self.games
        target = self.target_win_rate

        # Single line progress: Steps | Games | Win Rate | Target
        status = "✓" if win_rate >= target else " "
        print(
            f"  {self.num_timesteps:>8,} steps | {self.games:>4} games | "
            f"Win: {win_rate*100:>5.1f}% / {target*100:.0f}% {status}"
        )

        # Log to tensorboard silently
        if self.logger:
            self.logger.record("curriculum/win_rate", win_rate)
            self.logger.record("curriculum/games", self.games)

    def _on_training_end(self):
        """Called when training ends."""
        pass  # Summary handled by train_phase


class WinRateLoggingCallback(BaseCallback):
    """Silent callback to log win rates to tensorboard only."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_wins = []

    def _on_step(self) -> bool:
        # Track wins silently for tensorboard
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if (
                    info.get("player_score", 0) >= 3
                    or info.get("opponent_score", 0) >= 3
                ):
                    won = info.get("player_score", 0) > info.get("opponent_score", 0)
                    self.episode_wins.append(1.0 if won else 0.0)

                    # Log to tensorboard every 50 games
                    if len(self.episode_wins) % 50 == 0 and self.logger:
                        self.logger.record(
                            "rollout/win_rate", np.mean(self.episode_wins[-50:])
                        )

        return True


class RegressionDetectionCallback(BaseCallback):
    """
    Stops training if agent regresses on easy opponents.

    Periodically evaluates against SLOW_AI. If win rate drops below
    threshold, training is stopped to prevent catastrophic forgetting.
    Sets `regression_detected` flag so outer loop can also halt.
    """

    def __init__(
        self,
        check_freq: int = 50000,
        min_slow_ai_wr: float = 0.50,
        n_eval_episodes: int = 10,
        phase: int = 1,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.min_slow_ai_wr = min_slow_ai_wr
        self.n_eval_episodes = n_eval_episodes
        self.phase = phase
        self._last_check = 0
        self.regression_detected = False

    def _on_step(self) -> bool:
        # Only check in phases 2+ (phase 1 IS the easy opponent)
        if self.phase < 2:
            return True

        if self.num_timesteps - self._last_check >= self.check_freq:
            self._last_check = self.num_timesteps

            results = evaluate_model(
                self.model,
                OpponentType.SLOW_AI,
                ball_speed=0.6,
                n_episodes=self.n_eval_episodes,
                deterministic=False,
                verbose=False,
            )

            if results["win_rate"] < self.min_slow_ai_wr:
                print(
                    f"\n⚠️  REGRESSION DETECTED: {results['win_rate']*100:.0f}% vs SLOW_AI "
                    f"(< {self.min_slow_ai_wr*100:.0f}% threshold)"
                )
                print("  Stopping phase to prevent catastrophic forgetting.")
                self.regression_detected = True
                return False  # Stop current model.learn()

        return True


def make_ppo_env(
    opponent_type: OpponentType,
    ball_speed: float = 1.0,
    max_score: int = 3,
    max_steps: int = 5000,
    hit_reward: float = 1.0,
    tracking_reward: float = 0.01,
    pressure_scale: float = 0.0,
    step_penalty: float = 0.0,
) -> gym.Env:
    """
    Create a single Pong environment for PPO training.

    Uses offensive pressure shaping rewards. Hit rewards disabled by default.

    Args:
        opponent_type: Type of opponent AI
        ball_speed: Ball speed multiplier
        max_score: Points needed to win
        max_steps: Maximum steps per episode
        hit_reward: Reward for hitting ball (0 by default)
        tracking_reward: Reward for good positioning
        pressure_scale: Reward for pressuring opponent
        step_penalty: Per-step penalty

    Returns:
        Wrapped Pong environment
    """
    # Create base environment (headless, PPO-optimized)
    env = PongHeadlessEnv(
        ball_speed_multiplier=ball_speed,
        opponent_type=opponent_type,
        max_score=max_score,
        max_steps=max_steps,
    )

    # Apply reward wrapper
    env = WinFocusedRewardWrapper(
        env,
        point_reward=5.0,
        win_bonus=10.0,
        hit_reward=hit_reward,
        tracking_reward=tracking_reward,
        pressure_scale=pressure_scale,
        step_penalty=step_penalty,
    )

    # Add episode stats tracking
    env = EpisodeStatsWrapper(env)

    return env


def create_vec_env(
    opponent_type: OpponentType,
    ball_speed: float,
    n_envs: int = 4,
    use_subproc: bool = False,  # DummyVecEnv is more stable
    hit_reward: float = 1.0,
    tracking_reward: float = 0.01,
    pressure_scale: float = 0.0,
    step_penalty: float = 0.0,
) -> SubprocVecEnv:
    """
    Create vectorized environments for parallel training.

    Args:
        opponent_type: Type of opponent AI
        ball_speed: Ball speed multiplier
        n_envs: Number of parallel environments
        use_subproc: Use subprocess vectorization (faster but more memory)
        hit_reward: Reward for hitting ball
        tracking_reward: Reward for tracking ball
        pressure_scale: Reward for pressuring opponent
        step_penalty: Per-step penalty

    Returns:
        Vectorized environment
    """

    def make_env_fn() -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            env = make_ppo_env(
                opponent_type=opponent_type,
                ball_speed=ball_speed,
                max_score=3,  # Shorter games for faster training
                max_steps=5000,
                hit_reward=hit_reward,
                tracking_reward=tracking_reward,
                pressure_scale=pressure_scale,
                step_penalty=step_penalty,
            )
            env = Monitor(env)
            return env

        return _init

    env_fns = [make_env_fn() for _ in range(n_envs)]

    if use_subproc:
        return SubprocVecEnv(env_fns)
    else:
        return DummyVecEnv(env_fns)


def create_mixed_vec_env(
    opponent_mix: list,
    n_envs: int = 8,
    hit_reward: float = 0.05,
    tracking_reward: float = 0.01,
    pressure_scale: float = 0.08,
    step_penalty: float = 0.001,
) -> DummyVecEnv:
    """
    Create vectorized environments with mixed opponents.

    Distributes opponents across parallel envs to prevent catastrophic
    forgetting when training against hard opponents. E.g., 4 envs face
    REACTIVE_AI while 4 face NORMAL_AI/MEDIUM_AI for skill retention.

    Args:
        opponent_mix: List of (OpponentType, ball_speed, count) tuples.
            Sum of counts must equal n_envs.
        n_envs: Total parallel environments (must match sum of counts).
        hit_reward: Reward for hitting ball.
        tracking_reward: Reward for tracking ball.
        pressure_scale: Reward for pressuring opponent.
        step_penalty: Per-step penalty.

    Returns:
        DummyVecEnv with mixed opponents.
    """
    env_fns = []
    for opponent_type, ball_speed, count in opponent_mix:
        for _ in range(count):

            def make_fn(ot=opponent_type, bs=ball_speed):
                def _init():
                    env = make_ppo_env(
                        opponent_type=ot,
                        ball_speed=bs,
                        max_score=3,
                        max_steps=5000,
                        hit_reward=hit_reward,
                        tracking_reward=tracking_reward,
                        pressure_scale=pressure_scale,
                        step_penalty=step_penalty,
                    )
                    return Monitor(env)

                return _init

            env_fns.append(make_fn())

    assert len(env_fns) == n_envs, f"Mix counts ({len(env_fns)}) != n_envs ({n_envs})"
    return DummyVecEnv(env_fns)


def create_eval_env(opponent_type: OpponentType, ball_speed: float) -> gym.Env:
    """Create evaluation environment.

    Note: Stable-Baselines3 warns if training and eval env are different types
    (e.g., SubprocVecEnv vs DummyVecEnv). To avoid noisy warnings and keep
    behavior consistent, we return a VecEnv matching the training VecEnv type.
    """
    raise NotImplementedError("Use create_eval_vec_env() instead")


def create_eval_vec_env(
    opponent_type: OpponentType,
    ball_speed: float,
    use_subproc: bool,
    hit_reward: float = 1.0,
    tracking_reward: float = 0.01,
    pressure_scale: float = 0.0,
    step_penalty: float = 0.0,
) -> "SubprocVecEnv | DummyVecEnv":
    """Create evaluation VecEnv matching the training VecEnv type."""

    def _init() -> gym.Env:
        env = make_ppo_env(
            opponent_type=opponent_type,
            ball_speed=ball_speed,
            max_score=3,
            max_steps=5000,
            hit_reward=hit_reward,
            tracking_reward=tracking_reward,
            pressure_scale=pressure_scale,
            step_penalty=step_penalty,
        )
        return Monitor(env)

    if use_subproc:
        return SubprocVecEnv([_init])
    return DummyVecEnv([_init])


def train_phase(
    phase: int,
    model: Optional[PPO] = None,
    n_envs: int = 8,
    save_dir: str = "./models/",
    tensorboard_log: str = "./tensorboard/",
    verbose: int = 1,
    max_timesteps_multiplier: float = 5.0,
) -> Tuple[PPO, bool]:
    """
    Train one phase of the curriculum until win rate target is met.

    Args:
        phase: Phase number (1, 2, or 3)
        model: Existing model to continue training (None for new model)
        n_envs: Number of parallel environments
        save_dir: Directory to save models
        tensorboard_log: TensorBoard log directory
        verbose: Verbosity level
        max_timesteps_multiplier: Max timesteps = base × multiplier (safety cap)

    Returns:
        Tuple of (trained PPO model, whether target was achieved)
    """
    phase_config = CURRICULUM_PHASES[phase]
    base_timesteps = phase_config["timesteps"]
    max_timesteps = int(base_timesteps * max_timesteps_multiplier)

    print(f"\n{'='*50}")
    print(f"PHASE {phase}: {phase_config['name']}")
    print(f"{'='*50}")
    print(f"  Opponent: {phase_config['opponent_type'].value}")
    print(f"  Ball Speed: {phase_config['ball_speed']}x")
    print(f"  Target: {phase_config['target_win_rate']*100:.0f}% win rate")
    print(f"  LR: {phase_config['learning_rate']}, Entropy: {phase_config['ent_coef']}")
    print(f"  Max steps: {max_timesteps:,}")

    # Extract reward params from phase config
    reward_kwargs = {
        "hit_reward": phase_config.get("hit_reward", 0.0),
        "tracking_reward": phase_config.get("tracking_reward", 0.01),
        "pressure_scale": phase_config.get("pressure_scale", 0.02),
        "step_penalty": phase_config.get("step_penalty", 0.001),
    }

    # Create environments (DummyVecEnv is more stable than SubprocVecEnv)
    use_subproc = False

    if "mixed_opponents" in phase_config:
        # Mixed-opponent training: distribute different opponents across envs
        train_env = create_mixed_vec_env(
            opponent_mix=phase_config["mixed_opponents"],
            n_envs=n_envs,
            **reward_kwargs,
        )
    else:
        train_env = create_vec_env(
            opponent_type=phase_config["opponent_type"],
            ball_speed=phase_config["ball_speed"],
            n_envs=n_envs,
            use_subproc=use_subproc,
            **reward_kwargs,
        )

    eval_env = create_eval_vec_env(
        opponent_type=phase_config["opponent_type"],
        ball_speed=phase_config["ball_speed"],
        use_subproc=use_subproc,
        **reward_kwargs,
    )

    # Create or update model
    if model is None:
        print("Creating new model...")
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=phase_config["learning_rate"],
            n_steps=1024,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=phase_config["ent_coef"],
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,  # Disable table output
            tensorboard_log=tensorboard_log,
            device="auto",
        )
    else:
        print("Loading model from previous phase...")
        # Update hyperparameters for this phase (LR/entropy decay)
        # Assigning model.learning_rate alone has NO effect: Stable-Baselines3
        # builds self.lr_schedule once inside _setup_model(), and
        # _update_learning_rate() reads that schedule, not the attribute.
        # Rebuilding the schedule makes the new rate reach the optimizer,
        # because PPO.train() calls _update_learning_rate(self.policy.optimizer)
        # at the start of every update phase.
        model.learning_rate = phase_config["learning_rate"]
        model._setup_lr_schedule()
        model.ent_coef = phase_config["ent_coef"]
        model.set_env(train_env)

    # Training state
    start_time = time.time()
    total_timesteps_trained = 0
    target_achieved = False
    consecutive_target_hits = 0
    REQUIRED_CONSECUTIVE_HITS = 2  # Need to hit target 2 times in a row for early exit

    print(
        f"\nTraining until {phase_config['target_win_rate']*100:.0f}% win rate "
        f"(stable x{REQUIRED_CONSECUTIVE_HITS}) or {max_timesteps:,} steps max\n"
    )

    chunk_size = base_timesteps
    chunk_num = 0

    try:
        while total_timesteps_trained < max_timesteps and not target_achieved:
            chunk_num += 1
            remaining = max_timesteps - total_timesteps_trained
            this_chunk = min(chunk_size, remaining)

            print(f"--- Chunk {chunk_num} ({this_chunk:,} steps) ---")

            # Create callbacks (minimal, quiet)
            curriculum_callback = CurriculumCallback(
                phase=phase,
                target_win_rate=phase_config["target_win_rate"],
                log_freq=10000,
            )

            checkpoint_callback = CheckpointCallback(
                save_freq=50000,
                save_path=save_dir,
                name_prefix=f"ppo_phase{phase}",
                verbose=0,
            )

            # Silent eval callback - just saves best model
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(save_dir, f"best_phase{phase}"),
                log_path=os.path.join(save_dir, "logs"),
                eval_freq=25000,
                n_eval_episodes=10,
                deterministic=False,
                verbose=0,  # No output
            )

            win_rate_callback = WinRateLoggingCallback()

            # Check more frequently in later phases to catch regression early
            regression_freq = 50000 if phase <= 3 else 25000
            regression_callback = RegressionDetectionCallback(
                check_freq=regression_freq,
                min_slow_ai_wr=0.50,
                n_eval_episodes=10,
                phase=phase,
            )

            callbacks = CallbackList(
                [
                    curriculum_callback,
                    checkpoint_callback,
                    eval_callback,
                    win_rate_callback,
                    regression_callback,
                ]
            )

            # Train this chunk
            model.learn(
                total_timesteps=this_chunk,
                callback=callbacks,
                reset_num_timesteps=False,
                tb_log_name=f"phase{phase}",
            )

            total_timesteps_trained += this_chunk

            # Check if regression callback halted training
            if regression_callback.regression_detected:
                print("  Halting phase due to regression.")
                break

            # Evaluate after chunk. The gate is a decision, so it uses
            # deterministic (greedy) actions and enough episodes to be readable:
            # 100 episodes give a standard error near 4.6 percentage points at a
            # 30% win rate, against 10.2 points with 20 episodes. Episodes still
            # differ because the environment draws a new random ball launch on
            # every reset and is never re-seeded here.
            eval_results = evaluate_model(
                model,
                phase_config["opponent_type"],
                ball_speed=phase_config["ball_speed"],
                max_steps=5000,
                n_episodes=100,
                deterministic=True,
                verbose=False,
            )
            current_win_rate = eval_results["win_rate"]
            target_rate = phase_config["target_win_rate"]

            # Clean single-line evaluation result
            pct = total_timesteps_trained / max_timesteps * 100
            record = (
                f"{eval_results['wins']}W-{eval_results['losses']}L"
                f"-{eval_results['draws']}D, "
                f"{eval_results['unfinished']} unfinished"
            )

            if current_win_rate >= target_rate:
                consecutive_target_hits += 1
                status = f"✓ ({consecutive_target_hits}/{REQUIRED_CONSECUTIVE_HITS})"

                if consecutive_target_hits >= REQUIRED_CONSECUTIVE_HITS:
                    target_achieved = True
                    print(
                        f"\nEval: {current_win_rate*100:.0f}% win ({record}) {status}"
                    )
                    print(
                        f"\n✅ TARGET ACHIEVED after {total_timesteps_trained:,} steps"
                    )
                else:
                    print(f"Eval: {current_win_rate*100:.0f}% win ({record}) {status}")
            else:
                if consecutive_target_hits > 0:
                    status = "✗ reset"
                else:
                    status = ""
                consecutive_target_hits = 0
                print(
                    f"Eval: {current_win_rate*100:.0f}% win ({record}) [{pct:.0f}% of max] {status}"
                )

    except KeyboardInterrupt:
        print("\n\n⏹️  Training interrupted by user")

    elapsed = time.time() - start_time

    # Final evaluation: this sets target_achieved, which run_full_curriculum
    # uses to decide whether the next phase starts, so it is the same decision
    # as the per-chunk gate and uses the same settings (greedy actions,
    # 100 episodes).
    if not target_achieved:
        print(f"\nFinal evaluation for Phase {phase}...")
        final_eval = evaluate_model(
            model,
            phase_config["opponent_type"],
            ball_speed=phase_config["ball_speed"],
            max_steps=5000,
            n_episodes=100,
            deterministic=True,
            verbose=False,
        )
        final_win_rate = final_eval["win_rate"]
        target_rate = phase_config["target_win_rate"]

        if final_win_rate >= target_rate:
            target_achieved = True
            print(
                f"  ✅ Final: {final_win_rate*100:.0f}% win rate >= {target_rate*100:.0f}% target"
            )
        else:
            print(
                f"  Final: {final_win_rate*100:.0f}% win rate < {target_rate*100:.0f}% target"
            )

    # Clean phase summary
    print(
        f"\nPhase {phase} complete: {elapsed/60:.1f} min, {total_timesteps_trained:,} steps"
    )
    if not target_achieved:
        print(f"  Note: Target not achieved")

    # Save phase model
    phase_model_path = os.path.join(save_dir, f"ppo_phase{phase}_final")
    model.save(phase_model_path)
    print(f"  Saved: {phase_model_path}.zip")

    # Cleanup
    train_env.close()
    eval_env.close()

    return model, target_achieved


def evaluate_model(
    model: PPO,
    opponent_type: OpponentType,
    ball_speed: float = 1.0,
    max_score: int = 3,
    max_steps: int = 5000,
    n_episodes: int = 20,
    deterministic: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Evaluate model against specified opponent.

    Args:
        model: Trained PPO model
        opponent_type: Type of opponent to evaluate against
        ball_speed: Ball speed multiplier
        max_score: Points needed to win
        max_steps: Max steps per episode
        n_episodes: Number of evaluation episodes
        deterministic: Use deterministic actions (recommended for evaluation)
        verbose: Print per-episode results

    Returns:
        Dictionary with evaluation results
    """
    env = PongHeadlessEnv(
        opponent_type=opponent_type,
        ball_speed_multiplier=ball_speed,
        max_score=max_score,
        max_steps=max_steps,
    )

    wins = 0
    losses = 0
    draws = 0
    completed = 0  # episodes where somebody actually reached max_score
    completed_wins = 0
    total_score_diff = 0

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        terminated = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        player_score = info.get("player_score", 0)
        opponent_score = info.get("opponent_score", 0)

        # Three distinct outcomes by final score. Previously anything that was
        # not a win was reported as a loss, so a draw - including a 0-0 game
        # that simply ran out of steps - was indistinguishable from a defeat.
        if player_score > opponent_score:
            wins += 1
        elif opponent_score > player_score:
            losses += 1
        else:
            draws += 1

        # An episode only reached a real conclusion if it terminated. Hitting
        # the step cap means nobody reached max_score, whatever the score was.
        if terminated:
            completed += 1
            if player_score > opponent_score:
                completed_wins += 1

        total_score_diff += player_score - opponent_score

        if verbose:
            if player_score > opponent_score:
                result = "WIN"
            elif opponent_score > player_score:
                result = "LOSS"
            else:
                result = "DRAW"
            tag = "" if terminated else "  [unfinished]"
            print(
                f"  Episode {ep+1}/{n_episodes}: {result} ({player_score}-{opponent_score}){tag}"
            )

    env.close()

    return {
        "opponent": opponent_type.value,
        "ball_speed": ball_speed,
        "max_score": max_score,
        "max_steps": max_steps,
        "n_episodes": n_episodes,
        "deterministic": deterministic,
        "wins": wins,
        # Real defeats only. This used to be n_episodes - wins, which silently
        # folded draws and unfinished games into the loss count.
        "losses": losses,
        "draws": draws,
        "unfinished": n_episodes - completed,
        # win_rate keeps its original meaning - the share of episodes that
        # ended with the agent ahead on points - so phase gating and the
        # benchmark thresholds decide exactly what they decided before.
        "win_rate": wins / n_episodes,
        # Win rate over the episodes that actually finished. None when none of
        # them did, which is the signal that max_steps is too low rather than
        # that the agent is weak.
        "decided_win_rate": (completed_wins / completed) if completed else None,
        "unfinished_rate": (n_episodes - completed) / n_episodes,
        "avg_score_diff": total_score_diff / n_episodes,
    }


def run_full_curriculum(
    start_phase: int = 1,
    n_envs: int = 8,
    save_dir: str = "./models/",
    tensorboard_log: str = "./tensorboard/",
    require_target: bool = True,
) -> PPO:
    """
    Run the full 3-phase curriculum.

    Each phase must achieve its win rate target before advancing to the next.
    If a phase fails to meet its target (after max timesteps), training stops.

    Args:
        start_phase: Phase to start from (1, 2, or 3)
        n_envs: Number of parallel environments
        save_dir: Directory to save models
        tensorboard_log: TensorBoard log directory
        require_target: If True, must achieve target to advance (default: True)

    Returns:
        Final trained model
    """
    print("\n" + "=" * 50)
    print("PONG PPO CURRICULUM TRAINING")
    print("=" * 50)
    for phase_num, config in CURRICULUM_PHASES.items():
        marker = ">" if phase_num == start_phase else " "
        print(
            f" {marker} Phase {phase_num}: {config['name']} "
            f"({config['target_win_rate']*100:.0f}% target)"
        )

    if require_target:
        print("\nGated: must achieve target to advance")

    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(tensorboard_log, exist_ok=True)

    model = None

    # Check for existing model to resume from
    if start_phase > 1:
        prev_phase = start_phase - 1
        prev_model_path = os.path.join(save_dir, f"ppo_phase{prev_phase}_final.zip")
        if os.path.exists(prev_model_path):
            print(f"Loading model from Phase {prev_phase}: {prev_model_path}")
            model = PPO.load(prev_model_path)

    # Train each phase (3 phases total)
    total_start = time.time()
    final_phase_completed = 0

    for phase in range(start_phase, NUM_PHASES + 1):  # Phases 1, 2, 3
        model, target_achieved = train_phase(
            phase=phase,
            model=model,
            n_envs=n_envs,
            save_dir=save_dir,
            tensorboard_log=tensorboard_log,
        )

        final_phase_completed = phase

        # Gate: don't advance if target not achieved
        if require_target and not target_achieved:
            print(f"\n{'='*50}")
            print(f"STOPPED AT PHASE {phase}")
            print(f"Target not achieved after max steps.")
            print(f"{'='*50}")
            break

    total_time = time.time() - total_start

    print(f"\n{'='*50}")
    if final_phase_completed == NUM_PHASES:
        print(f"CURRICULUM COMPLETE")
    else:
        print(f"TRAINING ENDED (Phase {final_phase_completed})")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"{'='*50}")

    # Final evaluation - compact format
    print("\nFINAL RESULTS:")
    for opponent in [
        OpponentType.SLOW_AI,
        OpponentType.BEGINNER_AI,
        OpponentType.NORMAL_AI,
        OpponentType.REACTIVE_AI,
    ]:
        results = evaluate_model(
            model, opponent, n_episodes=20, deterministic=False, verbose=False
        )
        print(
            f"  vs {opponent.value}: {results['win_rate']*100:.0f}% "
            f"({results['wins']}W-{results['losses']}L-{results['draws']}D, "
            f"{results['unfinished']} unfinished)"
        )

    # Save final model
    final_path = os.path.join(save_dir, "ppo_final")
    model.save(final_path)
    print(f"\nSaved: {final_path}.zip")

    return model


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train Pong agent with PPO and 3-phase curriculum"
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Starting phase (1=Easy, 2=Beginner, 3=Medium, 4=Competitive, 5=Master)",
    )
    parser.add_argument(
        "--single-phase",
        action="store_true",
        help="Train only the specified phase, don't continue to subsequent phases",
    )
    parser.add_argument(
        "--envs",
        type=int,
        default=8,
        help="Number of parallel environments (default: 8)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./models/",
        help="Directory to save models",
    )
    parser.add_argument(
        "--tensorboard",
        type=str,
        default="./tensorboard/",
        help="TensorBoard log directory",
    )
    parser.add_argument(
        "--evaluate",
        type=str,
        default=None,
        help="Path to model to evaluate (skip training)",
    )

    args = parser.parse_args()

    # Evaluation mode
    if args.evaluate:
        print(f"Evaluating model: {args.evaluate}")
        model = PPO.load(args.evaluate)

        for opponent in [
            OpponentType.SLOW_AI,
            OpponentType.BEGINNER_AI,
            OpponentType.NORMAL_AI,
            OpponentType.REACTIVE_AI,
        ]:
            print(f"\nvs {opponent.value}:")
            results = evaluate_model(model, opponent, n_episodes=20)
            print(f"  Win Rate: {results['win_rate']*100:.1f}%")

        return

    # Training mode
    if args.single_phase:
        print(f"Training single phase: Phase {args.phase}")

        # Try to load previous phase model if not phase 1
        model = None
        if args.phase > 1:
            prev_path = os.path.join(
                args.save_dir, f"ppo_phase{args.phase-1}_final.zip"
            )
            if os.path.exists(prev_path):
                model = PPO.load(prev_path)
                print(f"Loaded model from: {prev_path}")

        model, target_achieved = train_phase(
            phase=args.phase,
            model=model,
            n_envs=args.envs,
            save_dir=args.save_dir,
            tensorboard_log=args.tensorboard,
        )

        if target_achieved:
            print(f"\n✅ Phase {args.phase} target achieved!")
        else:
            print(f"\n⚠️  Phase {args.phase} target NOT achieved")
    else:
        # Full curriculum
        run_full_curriculum(
            start_phase=args.phase,
            n_envs=args.envs,
            save_dir=args.save_dir,
            tensorboard_log=args.tensorboard,
        )


if __name__ == "__main__":
    main()
