"""
Gymnasium wrappers for Pong environment.

Provides reward shaping wrappers for different training objectives.
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any


class WinFocusedRewardWrapper(gym.Wrapper):
    """
    Reward wrapper focused on WINNING, not rallying.
    
    Offensive rewards (every step when ball moving toward opponent):
    - Pressure reward: positive when ball is heading where opponent ISN'T
    
    Defensive rewards (every step when ball moving toward player):
    - Tracking reward: small positive when paddle is aligned with ball
    
    Sparse rewards (on events):
    - +5 for scoring a point
    - -5 for opponent scoring
    - +10 bonus for winning the game
    - -10 penalty for losing the game
    
    Step penalty encourages faster games.
    Hit reward is DISABLED by default to prevent rally-based learning.
    """
    
    def __init__(
        self,
        env: gym.Env,
        point_reward: float = 5.0,      # Reward for scoring
        win_bonus: float = 10.0,        # Bonus for winning
        hit_reward: float = 0.0,        # DISABLED by default - prevents rally learning
        tracking_reward: float = 0.01,  # Tracking when ball approaches
        step_penalty: float = 0.0,      # No step penalty
        pressure_scale: float = 0.0,    # Disable pressure (simpler learning)
    ):
        """
        Initialize the reward wrapper.
        
        Args:
            env: The base Pong environment to wrap
            point_reward: Reward for scoring a point
            win_bonus: Bonus reward for winning the game
            hit_reward: Reward for hitting the ball (0 by default)
            tracking_reward: Per-step reward for good paddle positioning
            step_penalty: Per-step penalty to encourage faster games
            pressure_scale: Max reward for putting opponent under pressure
        """
        super().__init__(env)
        
        self.point_reward = point_reward
        self.win_bonus = win_bonus
        self.hit_reward = hit_reward
        self.tracking_reward = tracking_reward
        self.step_penalty = step_penalty
        self.pressure_scale = pressure_scale
        
        # Track state for reward calculation
        self._prev_player_score = 0
        self._prev_opponent_score = 0
        self._prev_ball_x = 0.5
        self._ball_approaching = False  # Ball moving toward player
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and tracking state."""
        obs, info = self.env.reset(**kwargs)
        
        self._prev_player_score = 0
        self._prev_opponent_score = 0
        self._prev_ball_x = obs[0] if len(obs) > 0 else 0.5
        self._ball_approaching = False
        
        return obs, info
    
    def _predict_ball_y_at_opponent(
        self, ball_x: float, ball_y: float, ball_vx: float, ball_vy: float
    ) -> float:
        """
        Predict ball's y-position when it reaches opponent's side (x=0).
        
        Uses simple reflection model for wall bounces.
        All values are normalized [0, 1].
        
        Args:
            ball_x: Current ball x position (1 = player side, 0 = opponent side)
            ball_y: Current ball y position
            ball_vx: Ball x velocity (negative = moving toward opponent)
            ball_vy: Ball y velocity
            
        Returns:
            Predicted y position at opponent's x, bounded [0, 1]
        """
        if ball_vx >= 0:
            # Ball not moving toward opponent
            return ball_y
        
        # Time to reach opponent side (x = 0)
        # ball_x + ball_vx * t = 0  =>  t = -ball_x / ball_vx
        if abs(ball_vx) < 1e-6:
            return ball_y
        
        t = -ball_x / ball_vx
        
        # Predicted y without bounces
        predicted_y = ball_y + ball_vy * t
        
        # Apply wall bounces (reflect off top/bottom walls at y=0 and y=1)
        # Use modular arithmetic with reflection
        while predicted_y < 0 or predicted_y > 1:
            if predicted_y < 0:
                predicted_y = -predicted_y  # Reflect off bottom
            if predicted_y > 1:
                predicted_y = 2 - predicted_y  # Reflect off top
        
        return np.clip(predicted_y, 0, 1)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step and compute reward with offensive focus."""
        obs, _, terminated, truncated, info = self.env.step(action)
        
        # Extract observation components
        # obs = [ball_x, ball_y, ball_vx, ball_vy, player_y, opponent_y, ball_dist, p_score, o_score]
        ball_x = obs[0]  # Normalized [0, 1], 1 = player side
        ball_y = obs[1]  # Normalized [0, 1]
        ball_vx = obs[2]  # Normalized velocity
        ball_vy = obs[3]  # Normalized velocity
        player_y = obs[4]  # Normalized [0, 1]
        opponent_y = obs[5]  # Normalized [0, 1]
        
        player_score = info.get("player_score", 0)
        opponent_score = info.get("opponent_score", 0)
        
        reward = 0.0
        
        # === OFFENSIVE REWARDS (when ball moving toward opponent) ===
        
        if ball_vx < 0 and self.pressure_scale > 0:
            # Ball is moving toward opponent - reward placing it where they aren't
            predicted_y = self._predict_ball_y_at_opponent(ball_x, ball_y, ball_vx, ball_vy)
            
            # Distance from opponent paddle to predicted intersection
            distance = abs(predicted_y - opponent_y)
            
            # Normalize distance (max meaningful distance is 0.5 in normalized coords)
            normalized_distance = min(distance / 0.5, 1.0)
            
            # Reward proportional to how far the opponent is from where ball will arrive
            reward += self.pressure_scale * normalized_distance
        
        # === DEFENSIVE REWARDS (when ball moving toward player) ===
        
        # 1. Tracking reward: reward when paddle is vertically aligned with ball
        #    Only when ball is approaching (ball_vx > 0 means moving toward player)
        if ball_vx > 0:  # Ball approaching player
            self._ball_approaching = True
            # How well aligned is the paddle with the ball?
            alignment = 1.0 - abs(ball_y - player_y)  # 1.0 = perfect, 0.0 = opposite ends
            reward += self.tracking_reward * alignment
        
        # 2. Hit reward: when ball was approaching and now moving away
        #    This means we successfully hit it (DISABLED by default)
        if self._ball_approaching and ball_vx < 0:
            reward += self.hit_reward
            self._ball_approaching = False
        
        # === SPARSE REWARDS (on events) ===
        
        # 3. Scoring rewards
        if player_score > self._prev_player_score:
            reward += self.point_reward
        
        if opponent_score > self._prev_opponent_score:
            reward -= self.point_reward
        
        # 4. Game end bonuses
        if terminated:
            if player_score > opponent_score:
                reward += self.win_bonus
            elif opponent_score > player_score:
                reward -= self.win_bonus
        
        # 5. Step penalty (encourages faster games)
        reward -= self.step_penalty
        
        # Update tracking state
        self._prev_player_score = player_score
        self._prev_opponent_score = opponent_score
        self._prev_ball_x = ball_x
        
        return obs, reward, terminated, truncated, info


class NormalizedRewardWrapper(gym.Wrapper):
    """
    Wrapper that normalizes rewards to a consistent scale.
    
    Useful for stabilizing PPO training when reward magnitudes vary.
    """
    
    def __init__(self, env: gym.Env, scale: float = 0.1):
        """
        Initialize normalized reward wrapper.
        
        Args:
            env: Environment to wrap
            scale: Multiplier for all rewards
        """
        super().__init__(env)
        self.scale = scale
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take step and scale reward."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward * self.scale, terminated, truncated, info


class EpisodeStatsWrapper(gym.Wrapper):
    """
    Wrapper that tracks episode statistics for logging.
    
    Adds detailed stats to the info dict at episode end.
    """
    
    def __init__(self, env: gym.Env):
        """Initialize stats tracking wrapper."""
        super().__init__(env)
        self._episode_reward = 0.0
        self._episode_length = 0
        self._points_scored = 0
        self._points_lost = 0
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and stats."""
        obs, info = self.env.reset(**kwargs)
        self._episode_reward = 0.0
        self._episode_length = 0
        self._points_scored = 0
        self._points_lost = 0
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Track stats on each step."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self._episode_reward += reward
        self._episode_length += 1
        
        # Track scoring (based on info from environment)
        player_score = info.get("player_score", 0)
        opponent_score = info.get("opponent_score", 0)
        
        if player_score > self._points_scored:
            self._points_scored = player_score
        if opponent_score > self._points_lost:
            self._points_lost = opponent_score
        
        # Add stats to info on episode end
        if terminated or truncated:
            info["episode_stats"] = {
                "reward": self._episode_reward,
                "length": self._episode_length,
                "points_scored": self._points_scored,
                "points_lost": self._points_lost,
                "won": player_score > opponent_score,
            }
        
        return obs, reward, terminated, truncated, info


def make_ppo_env(
    opponent_type: str = "normal_ai",
    ball_speed: float = 1.0,
    max_score: int = 5,
    max_steps: int = 10000,
    point_reward: float = 5.0,
    win_bonus: float = 10.0,
    hit_reward: float = 0.0,
    pressure_scale: float = 0.02,
    step_penalty: float = 0.001,
) -> gym.Env:
    """
    Create a Pong environment configured for PPO training.

    Uses offensive pressure shaping + sparse rewards for winning.
    Hit rewards are disabled by default to prevent rally-based learning.
    
    Args:
        opponent_type: Type of opponent AI
        ball_speed: Ball speed multiplier
        max_score: Points needed to win
        max_steps: Maximum steps per episode
        point_reward: Reward per point scored
        win_bonus: Bonus for winning game
        hit_reward: Reward for hitting ball (0 by default)
        pressure_scale: Reward for pressuring opponent
        step_penalty: Per-step penalty
        
    Returns:
        Wrapped Pong environment
    """
    from pong.env.pong_headless import PongHeadlessEnv, OpponentType
    
    # Map string to OpponentType
    opponent_mapping = {
        "slow_ai": OpponentType.SLOW_AI,
        "beginner_ai": OpponentType.BEGINNER_AI,
        "medium_ai": OpponentType.MEDIUM_AI,
        "normal_ai": OpponentType.NORMAL_AI,
        "reactive_ai": OpponentType.REACTIVE_AI,
        "agent": OpponentType.AGENT,
    }
    
    opp_type = opponent_mapping.get(opponent_type.lower(), OpponentType.NORMAL_AI)
    
    # Create base environment
    env = PongHeadlessEnv(
        ball_speed_multiplier=ball_speed,
        opponent_type=opp_type,
        max_score=max_score,
        max_steps=max_steps,
    )
    
    # Apply reward wrapper with offensive focus
    env = WinFocusedRewardWrapper(
        env,
        point_reward=point_reward,
        win_bonus=win_bonus,
        hit_reward=hit_reward,
        tracking_reward=0.01,
        step_penalty=step_penalty,
        pressure_scale=pressure_scale,
    )
    
    # Add episode stats tracking
    env = EpisodeStatsWrapper(env)
    
    return env
