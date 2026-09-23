"""
Headless Pong Environment for RL Training.

Pure Python implementation without pygame for fast headless training.
Optimized for PPO training with simple, sparse rewards.

Features:
- 5 opponent types: SLOW_AI, BEGINNER_AI, NORMAL_AI, REACTIVE_AI, AGENT
- Simple score-based rewards
- Gymnasium-compatible API
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any


class OpponentType(Enum):
    """Opponent difficulty levels with meaningful gaps between tiers."""

    SLOW_AI = "slow_ai"  # 40% speed, 35px dead zone
    BEGINNER_AI = "beginner_ai"  # 55% speed, 28px dead zone
    MEDIUM_AI = "medium_ai"  # 65% speed, 22px dead zone
    NORMAL_AI = "normal_ai"  # 70% speed, 20px dead zone
    REACTIVE_AI = "reactive_ai"  # 70% speed, 18px dead zone + late prediction
    AGENT = "agent"  # Self-play


@dataclass
class GameConfig:
    """Game configuration parameters."""

    screen_width: int = 960
    screen_height: int = 720
    offset: int = 20
    paddle_width: int = 10
    paddle_height: int = 100
    ball_size: int = 15
    base_ball_speed: float = 6.0
    base_paddle_speed: float = 8.0
    max_score: int = 3
    max_steps: int = 10000
    # Ball acceleration: horizontal speed increases by this factor per paddle
    # hit (classic Pong mechanic). Combined with the serve speed and
    # max_score above, measured so games reach max_score in ~2,800 steps
    # (well within a 5,000-step cap). The cap stays below the speed (~25
    # px/step) where the ball would tunnel through the paddle in one step.
    ball_accel_per_hit: float = 0.25  # 25% speed boost per hit
    ball_max_speed_mult: float = 4.0  # Cap at 4x initial launch speed
    # Edge-hit deflection: exponent > 1 makes paddle edges deflect
    # much more sharply than center (mimics original Atari Pong segments)
    edge_hit_exponent: float = 2.0  # Quadratic: edges deflect 2x-3x steeper
    edge_hit_scale: float = 3.0  # Max vy added on extreme edge hit


class PongHeadlessEnv(gym.Env):
    """
    Headless Pong environment for fast RL training with PPO.

    Observation space: 9-dimensional normalized vector
        [ball_x, ball_y, ball_vx, ball_vy, player_y, opponent_y, ball_dist, player_score, opp_score]

    Action space: Discrete(3) - 0: stay, 1: up, 2: down

    Rewards: Simple score-based (+1 for scoring, -1 for opponent scoring)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        ball_speed_multiplier: float = 1.0,
        opponent_type: OpponentType = OpponentType.NORMAL_AI,
        agent_controlled_opponent: bool = False,
        max_score: Optional[int] = None,
        max_steps: Optional[int] = None,
        config: Optional[GameConfig] = None,
    ):
        """
        Initialize headless Pong environment.

        Args:
            render_mode: Rendering mode ('human', 'rgb_array', or None)
            ball_speed_multiplier: Ball speed multiplier
            opponent_type: Type of opponent AI
            agent_controlled_opponent: If True, opponent uses agent actions
            max_score: Points needed to win (overrides config)
            max_steps: Maximum steps per episode (overrides config)
            config: Game configuration
        """
        super().__init__()

        self.render_mode = render_mode
        self.ball_speed_multiplier = ball_speed_multiplier
        self.opponent_type = opponent_type
        self.agent_controlled_opponent = agent_controlled_opponent
        self.config = config or GameConfig()

        # Optional overrides
        if max_score is not None:
            self.config.max_score = int(max_score)
        if max_steps is not None:
            self.config.max_steps = int(max_steps)

        # Gymnasium spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(9,), dtype=np.float32
        )

        # Game state (will be initialized in reset())
        self._initialize_game_state()

        # Metrics tracking
        self._reset_metrics()

        # Opponent action buffer for self-play
        self._opponent_action = 0

    def _get_opponent_config(self) -> Tuple[float, float]:
        """
        Get opponent configuration (speed_multiplier, dead_zone).

        Returns:
            Tuple of (speed_multiplier, dead_zone)
        """
        configs = {
            OpponentType.SLOW_AI: (0.40, 35.0),
            OpponentType.BEGINNER_AI: (0.55, 28.0),
            OpponentType.MEDIUM_AI: (0.65, 22.0),
            OpponentType.NORMAL_AI: (0.70, 20.0),
            OpponentType.REACTIVE_AI: (
                0.70,
                18.0,
            ),  # Same speed as NORMAL + delayed prediction
            OpponentType.AGENT: (1.0, 0.0),
        }
        return configs.get(self.opponent_type, (0.85, 15.0))

    def _initialize_game_state(self):
        """Initialize all game state variables."""
        c = self.config

        # Ball state
        self.ball_x = c.screen_width / 2
        self.ball_y = c.screen_height / 2
        self.ball_vx = 0.0
        self.ball_vy = 0.0

        # Paddle positions (y-center)
        self.player_y = c.screen_height / 2
        self.opponent_y = c.screen_height / 2

        # Player paddle on RIGHT side
        self.player_x = c.screen_width - c.offset - c.paddle_width
        # Opponent paddle on LEFT side
        self.opponent_x = c.offset + c.paddle_width

        # Scores
        self.player_score = 0
        self.opponent_score = 0

        # Step counter
        self._steps = 0

    def _reset_metrics(self):
        """Reset episode metrics tracking."""
        self._hits = 0
        self._misses = 0
        self._rally_count = 0
        self._rally_length_sum = 0
        self._rally_length_count = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        # Reset game state
        self._initialize_game_state()
        self._reset_metrics()

        # Launch ball with random direction
        self._launch_ball()

        return self._get_obs(), self._get_info()

    def _launch_ball(self):
        """Launch ball from center with random direction."""
        c = self.config

        # Reset position
        self.ball_x = c.screen_width / 2
        self.ball_y = c.screen_height / 2

        # Random direction
        angle = self.np_random.uniform(-0.5, 0.5)  # Radians
        direction = self.np_random.choice([-1, 1])

        speed = c.base_ball_speed * self.ball_speed_multiplier
        self.ball_vx = direction * speed * np.cos(angle)
        self.ball_vy = speed * np.sin(angle)

        self._rally_count = 0

    def _get_obs(self) -> np.ndarray:
        """Get normalized observation vector."""
        c = self.config

        # Normalize positions to [0, 1]
        ball_x_norm = self.ball_x / c.screen_width
        ball_y_norm = self.ball_y / c.screen_height
        player_y_norm = self.player_y / c.screen_height
        opponent_y_norm = self.opponent_y / c.screen_height

        # Normalize velocities to [-1, 1]. The scale must cover the fastest
        # possible ball: launch speed at multiplier 1.0, times the
        # acceleration cap.
        max_speed = c.base_ball_speed * c.ball_max_speed_mult
        ball_vx_norm = np.clip(self.ball_vx / max_speed, -1.0, 1.0)
        ball_vy_norm = np.clip(self.ball_vy / max_speed, -1.0, 1.0)

        # Ball distance from player paddle (normalized)
        max_dist = np.sqrt(c.screen_width**2 + c.screen_height**2)
        ball_dist = np.sqrt(
            (self.ball_x - self.player_x) ** 2 + (self.ball_y - self.player_y) ** 2
        )
        ball_dist_norm = ball_dist / max_dist

        # Scores (normalized to [0, 1])
        player_score_norm = self.player_score / c.max_score
        opponent_score_norm = self.opponent_score / c.max_score

        return np.array(
            [
                ball_x_norm,
                ball_y_norm,
                ball_vx_norm,
                ball_vy_norm,
                player_y_norm,
                opponent_y_norm,
                ball_dist_norm,
                player_score_norm,
                opponent_score_norm,
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> Dict[str, Any]:
        """Get info dict with game metrics."""
        # Calculate metrics
        total_shots = self._hits + self._misses
        hit_rate = self._hits / max(total_shots, 1)

        avg_rally = (
            (self._rally_length_sum / self._rally_length_count)
            if self._rally_length_count
            else 0.0
        )

        # Win rate: 1.0 if agent won the game, 0.0 otherwise
        win_rate = 1.0 if self.player_score > self.opponent_score else 0.0

        return {
            "player_score": self.player_score,
            "opponent_score": self.opponent_score,
            "hit_rate": hit_rate,
            "avg_rally": avg_rally,
            "win_rate": win_rate,
            "steps": self._steps,
            "ball_position": (self.ball_x, self.ball_y),
            "hits": self._hits,
            "misses": self._misses,
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take one step in the environment.

        Args:
            action: 0 (stay), 1 (up), 2 (down)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        c = self.config
        self._steps += 1

        # Store previous state for reward calculation
        prev_player_score = self.player_score
        prev_opponent_score = self.opponent_score

        # Move player paddle
        self._move_player(action)

        # Move opponent paddle
        self._move_opponent()

        # Update ball position
        self._update_ball()

        # Check collisions
        self._check_collisions()

        # Check scoring
        self._check_scoring()

        # Calculate simple score-based reward
        reward = 0.0
        if self.player_score > prev_player_score:
            reward += 1.0  # Scored a point
        if self.opponent_score > prev_opponent_score:
            reward -= 1.0  # Lost a point

        # Check termination
        terminated = (
            self.player_score >= c.max_score or self.opponent_score >= c.max_score
        )

        truncated = self._steps >= c.max_steps

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _move_player(self, action: int):
        """Move player paddle based on action."""
        c = self.config

        if action == 1:  # Up
            self.player_y -= c.base_paddle_speed
        elif action == 2:  # Down
            self.player_y += c.base_paddle_speed

        # Constrain to screen bounds
        half_paddle = c.paddle_height / 2
        self.player_y = np.clip(
            self.player_y,
            c.offset + half_paddle,
            c.screen_height - c.offset - half_paddle,
        )

    def _move_opponent(self):
        """Move opponent paddle based on AI type."""
        if self.agent_controlled_opponent:
            self._move_agent_opponent()
        else:
            self._move_ai_opponent()

    def _move_ai_opponent(self):
        """Move opponent using AI logic."""
        c = self.config
        speed_mult, dead_zone = self._get_opponent_config()

        # Target position - default is ball tracking
        target_y = self.ball_y

        # REACTIVE_AI uses late prediction - only activates when ball
        # is in the last 30% of the court (near opponent). This gives
        # the agent's side shots time to create angles before the
        # opponent starts predicting the trajectory.
        if self.opponent_type == OpponentType.REACTIVE_AI and self.ball_vx < 0:
            prediction_zone = c.screen_width * 0.30  # Last 30% of court
            if self.ball_x < prediction_zone and abs(self.ball_vx) > 0.1:
                time_to_reach = (self.ball_x - self.opponent_x) / abs(self.ball_vx)
                if time_to_reach > 0:
                    predicted_y = self.ball_y + self.ball_vy * time_to_reach
                    # Reflect off walls
                    min_y = c.offset + c.ball_size / 2
                    max_y = c.screen_height - c.offset - c.ball_size / 2
                    while predicted_y < min_y or predicted_y > max_y:
                        if predicted_y < min_y:
                            predicted_y = 2 * min_y - predicted_y
                        if predicted_y > max_y:
                            predicted_y = 2 * max_y - predicted_y
                    target_y = predicted_y

        # Move toward target with dead zone
        diff = target_y - self.opponent_y
        speed = c.base_paddle_speed * speed_mult

        if abs(diff) > dead_zone:
            if diff > 0:
                self.opponent_y += speed
            else:
                self.opponent_y -= speed

        # Constrain to screen bounds
        half_paddle = c.paddle_height / 2
        self.opponent_y = np.clip(
            self.opponent_y,
            c.offset + half_paddle,
            c.screen_height - c.offset - half_paddle,
        )

    def _move_agent_opponent(self):
        """Move opponent using agent action."""
        c = self.config

        if self._opponent_action == 1:  # Up
            self.opponent_y -= c.base_paddle_speed
        elif self._opponent_action == 2:  # Down
            self.opponent_y += c.base_paddle_speed

        # Constrain to screen bounds
        half_paddle = c.paddle_height / 2
        self.opponent_y = np.clip(
            self.opponent_y,
            c.offset + half_paddle,
            c.screen_height - c.offset - half_paddle,
        )

    def set_opponent_action(self, action: int):
        """Set the action for agent-controlled opponent."""
        self._opponent_action = action

    def _update_ball(self):
        """Update ball position."""
        c = self.config

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # Wall bounces (top and bottom)
        if self.ball_y <= c.offset + c.ball_size / 2:
            self.ball_y = c.offset + c.ball_size / 2
            self.ball_vy = abs(self.ball_vy)
        elif self.ball_y >= c.screen_height - c.offset - c.ball_size / 2:
            self.ball_y = c.screen_height - c.offset - c.ball_size / 2
            self.ball_vy = -abs(self.ball_vy)

    def _apply_paddle_bounce(self, paddle_y: float, direction: int):
        """
        Apply paddle bounce with edge-amplified deflection and acceleration.

        Mimics original Atari Pong: center hits bounce nearly straight,
        but the paddle tips produce dramatically steeper angles via a
        power-curve deflection. Horizontal speed accelerates per hit.

        Args:
            paddle_y: Y-center of the paddle that was hit
            direction: 1 for rightward (opponent hit), -1 for leftward (player hit)
        """
        c = self.config
        half_paddle = c.paddle_height / 2

        # Compute hit offset: -1.0 (top edge) to +1.0 (bottom edge)
        offset = np.clip((self.ball_y - paddle_y) / half_paddle, -1.0, 1.0)

        # Accelerate horizontal speed per hit (capped at max)
        launch_speed = c.base_ball_speed * self.ball_speed_multiplier
        max_hspeed = launch_speed * c.ball_max_speed_mult
        new_hspeed = min(abs(self.ball_vx) * (1.0 + c.ball_accel_per_hit), max_hspeed)

        # Set horizontal direction
        self.ball_vx = direction * new_hspeed

        # Edge-amplified vertical deflection (power curve)
        # Center hits (offset≈0) → mild deflection
        # Edge hits (offset≈±1) → sharp deflection (like original Pong tips)
        deflection = (
            np.sign(offset) * (abs(offset) ** c.edge_hit_exponent) * c.edge_hit_scale
        )
        self.ball_vy += deflection

    def _check_collisions(self):
        """Check and handle paddle collisions."""
        c = self.config
        half_paddle = c.paddle_height / 2
        ball_radius = c.ball_size / 2

        # Player paddle collision (right side)
        if self.ball_vx > 0:  # Ball moving right
            if (
                self.ball_x + ball_radius >= self.player_x - c.paddle_width / 2
                and self.ball_x - ball_radius <= self.player_x + c.paddle_width / 2
            ):
                if abs(self.ball_y - self.player_y) <= half_paddle + ball_radius:
                    # Hit!
                    self._hits += 1
                    self._rally_count += 1
                    self._apply_paddle_bounce(self.player_y, direction=-1)

        # Opponent paddle collision (left side)
        if self.ball_vx < 0:  # Ball moving left
            if (
                self.ball_x - ball_radius <= self.opponent_x + c.paddle_width / 2
                and self.ball_x + ball_radius >= self.opponent_x - c.paddle_width / 2
            ):
                if abs(self.ball_y - self.opponent_y) <= half_paddle + ball_radius:
                    # Opponent hit
                    self._rally_count += 1
                    self._apply_paddle_bounce(self.opponent_y, direction=1)

    def _check_scoring(self):
        """Check if a point was scored."""
        c = self.config

        # Ball past right edge (opponent scores)
        if self.ball_x > c.screen_width - c.offset:
            self.opponent_score += 1
            self._misses += 1

            # Save rally length
            if self._rally_count > 0:
                self._rally_length_sum += self._rally_count
                self._rally_length_count += 1

            self._launch_ball()

        # Ball past left edge (player scores)
        elif self.ball_x < c.offset:
            self.player_score += 1

            # Save rally length
            if self._rally_count > 0:
                self._rally_length_sum += self._rally_count
                self._rally_length_count += 1

            self._launch_ball()

    def render(self):
        """Render the environment (placeholder for headless)."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None

    def _render_frame(self) -> np.ndarray:
        """Render a frame as RGB array."""
        c = self.config

        # Create simple RGB frame
        frame = np.zeros((c.screen_height, c.screen_width, 3), dtype=np.uint8)

        # Draw paddles (white)
        paddle_half = c.paddle_height // 2
        paddle_w = c.paddle_width

        # Player paddle (right)
        p_top = int(self.player_y - paddle_half)
        p_bottom = int(self.player_y + paddle_half)
        p_left = int(self.player_x - paddle_w // 2)
        p_right = int(self.player_x + paddle_w // 2)
        frame[p_top:p_bottom, p_left:p_right] = [255, 255, 255]

        # Opponent paddle (left)
        o_top = int(self.opponent_y - paddle_half)
        o_bottom = int(self.opponent_y + paddle_half)
        o_left = int(self.opponent_x - paddle_w // 2)
        o_right = int(self.opponent_x + paddle_w // 2)
        frame[o_top:o_bottom, o_left:o_right] = [255, 255, 255]

        # Draw ball (white circle approximation)
        ball_radius = c.ball_size // 2
        bx, by = int(self.ball_x), int(self.ball_y)
        for dy in range(-ball_radius, ball_radius + 1):
            for dx in range(-ball_radius, ball_radius + 1):
                if dx * dx + dy * dy <= ball_radius * ball_radius:
                    px, py = bx + dx, by + dy
                    if 0 <= px < c.screen_width and 0 <= py < c.screen_height:
                        frame[py, px] = [255, 255, 255]

        return frame

    def close(self):
        """Clean up resources."""
        pass


def register_headless_env():
    """Register the headless Pong environment with gymnasium."""
    try:
        gym.register(
            id="PongHeadless-v0",
            entry_point="pong.env.pong_headless:PongHeadlessEnv",
            max_episode_steps=10000,
        )
    except gym.error.Error:
        # Already registered
        pass
