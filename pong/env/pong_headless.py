"""
Headless Pong Environment for RL Training.

Pure Python implementation without pygame for fast headless training.
The same trajectory and rebound model is shared with the graphical game.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from pong.game.physics import bounce_velocity, predict_intercept

# Optional extra observation features the RL coach can enable. Each name appends
# one dimension to the observation vector, in the order given. Enabling any of
# these changes the observation shape, so saved models are not compatible.
SUPPORTED_OBS_FEATURES = ("paddle_velocity", "intercept_y", "relative_offset")


class OpponentType(Enum):
    """Opponent difficulty levels with meaningful gaps between tiers."""

    SLOW_AI = "slow_ai"
    BEGINNER_AI = "beginner_ai"
    MEDIUM_AI = "medium_ai"
    NORMAL_AI = "normal_ai"
    REACTIVE_AI = "reactive_ai"
    EXPERT_AI = "expert_ai"
    AGENT = "agent"


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
    ball_accel_per_hit: float = 0.12
    ball_max_speed_mult: float = 3.0
    max_bounce_angle: float = 60.0
    paddle_spin_angle: float = 12.0
    obs_features: Tuple[str, ...] = ()


class PongHeadlessEnv(gym.Env):
    """
    Headless Pong environment for fast RL training with PPO.

    Observation space: 9-dimensional normalized vector
        [ball_x, ball_y, ball_vx, ball_vy, player_y, opponent_y, ball_dist, player_score, opp_score]
        plus one dimension per entry in config.obs_features (see
        SUPPORTED_OBS_FEATURES).

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
        obs_features: Optional[List[str]] = None,
    ):
        """Initialize the headless Pong environment."""
        super().__init__()

        self.render_mode = render_mode
        self.ball_speed_multiplier = ball_speed_multiplier
        self.opponent_type = opponent_type
        self.agent_controlled_opponent = agent_controlled_opponent
        self.config = config or GameConfig()

        if obs_features is not None:
            unknown = set(obs_features) - set(SUPPORTED_OBS_FEATURES)
            if unknown:
                raise ValueError(
                    f"Unknown obs_features {sorted(unknown)}; "
                    f"supported: {list(SUPPORTED_OBS_FEATURES)}"
                )
            self.config.obs_features = tuple(obs_features)

        if max_score is not None:
            self.config.max_score = int(max_score)
        if max_steps is not None:
            self.config.max_steps = int(max_steps)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(9 + len(self.config.obs_features),),
            dtype=np.float32,
        )

        self._initialize_game_state()
        self._reset_metrics()
        self._opponent_action = 0

    def _get_opponent_config(self) -> Tuple[float, float]:
        """Return opponent speed multiplier and movement dead zone."""
        configs = {
            OpponentType.SLOW_AI: (0.40, 35.0),
            OpponentType.BEGINNER_AI: (0.55, 28.0),
            OpponentType.MEDIUM_AI: (0.65, 22.0),
            OpponentType.NORMAL_AI: (0.70, 20.0),
            OpponentType.REACTIVE_AI: (0.70, 18.0),
            OpponentType.EXPERT_AI: (1.0, 1.0),
            OpponentType.AGENT: (1.0, 0.0),
        }
        return configs.get(self.opponent_type, (0.85, 15.0))

    def _initialize_game_state(self):
        """Initialize all game state variables."""
        c = self.config
        self.ball_x = c.screen_width / 2
        self.ball_y = c.screen_height / 2
        self.ball_vx = 0.0
        self.ball_vy = 0.0

        self.player_y = c.screen_height / 2
        self.opponent_y = c.screen_height / 2
        self.player_x = c.screen_width - c.offset - c.paddle_width
        self.opponent_x = c.offset + c.paddle_width

        self.player_score = 0
        self.opponent_score = 0
        self._steps = 0
        self._player_vy = 0.0
        self._opponent_vy = 0.0

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
        self._initialize_game_state()
        self._reset_metrics()
        self._launch_ball()
        return self._get_obs(), self._get_info()

    def _launch_ball(self):
        """Launch the ball from center at a playable random angle."""
        c = self.config
        self.ball_x = c.screen_width / 2
        self.ball_y = c.screen_height / 2
        angle = self.np_random.uniform(-0.5, 0.5)
        direction = self.np_random.choice([-1, 1])
        speed = c.base_ball_speed * self.ball_speed_multiplier
        self.ball_vx = direction * speed * np.cos(angle)
        self.ball_vy = speed * np.sin(angle)
        self._rally_count = 0

    def _get_obs(self) -> np.ndarray:
        """Get normalized observation vector."""
        c = self.config
        ball_x_norm = self.ball_x / c.screen_width
        ball_y_norm = self.ball_y / c.screen_height
        player_y_norm = self.player_y / c.screen_height
        opponent_y_norm = self.opponent_y / c.screen_height

        max_speed = c.base_ball_speed * c.ball_max_speed_mult
        ball_vx_norm = np.clip(self.ball_vx / max_speed, -1.0, 1.0)
        ball_vy_norm = np.clip(self.ball_vy / max_speed, -1.0, 1.0)

        max_dist = np.sqrt(c.screen_width**2 + c.screen_height**2)
        ball_dist = np.sqrt(
            (self.ball_x - self.player_x) ** 2 + (self.ball_y - self.player_y) ** 2
        )
        ball_dist_norm = ball_dist / max_dist

        obs = [
            ball_x_norm,
            ball_y_norm,
            ball_vx_norm,
            ball_vy_norm,
            player_y_norm,
            opponent_y_norm,
            ball_dist_norm,
            self.player_score / c.max_score,
            self.opponent_score / c.max_score,
        ]
        for feature in c.obs_features:
            obs.append(self._extra_feature(feature))
        return np.array(obs, dtype=np.float32)

    def _extra_feature(self, name: str) -> float:
        """Compute one optional observation feature."""
        c = self.config
        if name == "paddle_velocity":
            return float(np.clip(self._player_vy / c.base_paddle_speed, -1.0, 1.0))
        if name == "intercept_y":
            if self.ball_vx <= 0.1:
                return float(self.ball_y / c.screen_height)
            target_x = self.player_x - c.paddle_width / 2 - c.ball_size / 2
            predicted_y, _ = predict_intercept(
                self.ball_x,
                self.ball_y,
                self.ball_vx,
                self.ball_vy,
                target_x,
                c.offset + c.ball_size / 2,
                c.screen_height - c.offset - c.ball_size / 2,
            )
            return float(np.clip(predicted_y / c.screen_height, 0.0, 1.0))
        if name == "relative_offset":
            return float(
                np.clip((self.player_y - self.ball_y) / c.screen_height, -1.0, 1.0)
            )
        raise ValueError(f"Unknown obs feature: {name}")

    def _get_info(self) -> Dict[str, Any]:
        """Get info dict with game metrics."""
        total_shots = self._hits + self._misses
        avg_rally = (
            self._rally_length_sum / self._rally_length_count
            if self._rally_length_count
            else 0.0
        )
        return {
            "player_score": self.player_score,
            "opponent_score": self.opponent_score,
            "hit_rate": self._hits / max(total_shots, 1),
            "avg_rally": avg_rally,
            "win_rate": 1.0 if self.player_score > self.opponent_score else 0.0,
            "steps": self._steps,
            "ball_position": (self.ball_x, self.ball_y),
            "hits": self._hits,
            "misses": self._misses,
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take one fixed-duration simulation step."""
        c = self.config
        self._steps += 1
        prev_player_score = self.player_score
        prev_opponent_score = self.opponent_score

        self._move_player(action)
        self._move_opponent()
        self._update_ball()
        self._check_scoring()

        reward = 0.0
        if self.player_score > prev_player_score:
            reward += 1.0
        if self.opponent_score > prev_opponent_score:
            reward -= 1.0

        terminated = (
            self.player_score >= c.max_score or self.opponent_score >= c.max_score
        )
        truncated = self._steps >= c.max_steps
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _move_player(self, action: int):
        """Move player paddle based on action."""
        c = self.config
        prev_y = self.player_y
        if action == 1:
            self.player_y -= c.base_paddle_speed
        elif action == 2:
            self.player_y += c.base_paddle_speed
        half_paddle = c.paddle_height / 2
        self.player_y = np.clip(
            self.player_y,
            c.offset + half_paddle,
            c.screen_height - c.offset - half_paddle,
        )
        self._player_vy = self.player_y - prev_y

    def _move_opponent(self):
        """Move opponent paddle based on AI type."""
        if self.agent_controlled_opponent:
            self._move_agent_opponent()
        else:
            self._move_ai_opponent()

    def _predicted_target(self, paddle: str) -> float:
        """Return an intercept target with an attacking contact offset."""
        c = self.config
        is_player = paddle == "player"
        incoming = self.ball_vx > 0 if is_player else self.ball_vx < 0
        if not incoming:
            return c.screen_height / 2

        if is_player:
            target_x = self.player_x - c.paddle_width / 2 - c.ball_size / 2
            other_y = self.opponent_y
        else:
            target_x = self.opponent_x + c.paddle_width / 2 + c.ball_size / 2
            other_y = self.player_y

        intercept_y, _ = predict_intercept(
            self.ball_x,
            self.ball_y,
            self.ball_vx,
            self.ball_vy,
            target_x,
            c.offset + c.ball_size / 2,
            c.screen_height - c.offset - c.ball_size / 2,
        )
        contact_offset = 0.85 if other_y < c.screen_height / 2 else -0.85
        target_y = intercept_y - contact_offset * c.paddle_height / 2
        half_paddle = c.paddle_height / 2
        return float(
            np.clip(
                target_y,
                c.offset + half_paddle,
                c.screen_height - c.offset - half_paddle,
            )
        )

    def expert_action(self, paddle: str = "player") -> int:
        """Return the predictive controller's action for either paddle."""
        if paddle not in ("player", "opponent"):
            raise ValueError("paddle must be 'player' or 'opponent'")
        paddle_y = self.player_y if paddle == "player" else self.opponent_y
        difference = self._predicted_target(paddle) - paddle_y
        if abs(difference) <= 1.0:
            return 0
        return 2 if difference > 0 else 1

    def _move_ai_opponent(self):
        """Move opponent using tracking, prediction, or expert shot placement."""
        c = self.config
        speed_mult, dead_zone = self._get_opponent_config()
        previous_y = self.opponent_y

        if self.opponent_type == OpponentType.EXPERT_AI:
            target_y = self._predicted_target("opponent")
        else:
            target_y = self.ball_y
            if self.opponent_type == OpponentType.REACTIVE_AI and self.ball_vx < 0:
                if self.ball_x < c.screen_width * 0.30 and abs(self.ball_vx) > 0.1:
                    target_x = self.opponent_x + c.paddle_width / 2 + c.ball_size / 2
                    target_y, _ = predict_intercept(
                        self.ball_x,
                        self.ball_y,
                        self.ball_vx,
                        self.ball_vy,
                        target_x,
                        c.offset + c.ball_size / 2,
                        c.screen_height - c.offset - c.ball_size / 2,
                    )

        difference = target_y - self.opponent_y
        speed = c.base_paddle_speed * speed_mult
        if abs(difference) > dead_zone:
            self.opponent_y += np.clip(difference, -speed, speed)

        half_paddle = c.paddle_height / 2
        self.opponent_y = np.clip(
            self.opponent_y,
            c.offset + half_paddle,
            c.screen_height - c.offset - half_paddle,
        )
        self._opponent_vy = self.opponent_y - previous_y

    def _move_agent_opponent(self):
        """Move opponent using agent action."""
        c = self.config
        previous_y = self.opponent_y
        if self._opponent_action == 1:
            self.opponent_y -= c.base_paddle_speed
        elif self._opponent_action == 2:
            self.opponent_y += c.base_paddle_speed
        half_paddle = c.paddle_height / 2
        self.opponent_y = np.clip(
            self.opponent_y,
            c.offset + half_paddle,
            c.screen_height - c.offset - half_paddle,
        )
        self._opponent_vy = self.opponent_y - previous_y

    def set_opponent_action(self, action: int):
        """Set the action for agent-controlled opponent."""
        self._opponent_action = action

    def _apply_paddle_bounce(
        self, paddle_y: float, direction: int, paddle_velocity: float = 0.0
    ):
        """Apply an angle-preserving rebound with bounded speed and spin."""
        c = self.config
        hit_offset = (self.ball_y - paddle_y) / (
            c.paddle_height / 2 + c.ball_size / 2
        )
        launch_speed = c.base_ball_speed * self.ball_speed_multiplier
        self.ball_vx, self.ball_vy = bounce_velocity(
            self.ball_vx,
            self.ball_vy,
            hit_offset,
            direction,
            paddle_velocity,
            c.base_paddle_speed,
            c.ball_accel_per_hit,
            launch_speed * c.ball_max_speed_mult,
            c.max_bounce_angle,
            c.paddle_spin_angle,
        )

    def _update_ball(self):
        """Advance through collisions in time order so fast balls cannot tunnel."""
        c = self.config
        radius = c.ball_size / 2
        half_paddle = c.paddle_height / 2
        half_width = c.paddle_width / 2
        min_y = c.offset + radius
        max_y = c.screen_height - c.offset - radius
        right_plane = self.player_x - half_width - radius
        left_plane = self.opponent_x + half_width + radius
        right_back = self.player_x + half_width + radius
        left_back = self.opponent_x - half_width - radius
        remaining = 1.0
        missed_paddles = set()
        epsilon = 1e-9

        for _ in range(12):
            if remaining <= epsilon:
                break
            events = []

            if self.ball_vy < 0:
                events.append(((min_y - self.ball_y) / self.ball_vy, "wall"))
            elif self.ball_vy > 0:
                events.append(((max_y - self.ball_y) / self.ball_vy, "wall"))

            if self.ball_vx > 0:
                events.append(
                    ((c.screen_width - c.offset - self.ball_x) / self.ball_vx, "goal_right")
                )
                if "player" not in missed_paddles and self.ball_x <= right_back + epsilon:
                    hit_time = (right_plane - self.ball_x) / self.ball_vx
                    if hit_time < 0 and self.ball_x >= right_plane - epsilon:
                        hit_time = 0.0
                    if hit_time >= 0:
                        events.append((hit_time, "player"))
            elif self.ball_vx < 0:
                events.append(((c.offset - self.ball_x) / self.ball_vx, "goal_left"))
                if "opponent" not in missed_paddles and self.ball_x >= left_back - epsilon:
                    hit_time = (left_plane - self.ball_x) / self.ball_vx
                    if hit_time < 0 and self.ball_x <= left_plane + epsilon:
                        hit_time = 0.0
                    if hit_time >= 0:
                        events.append((hit_time, "opponent"))

            valid_events = [
                (time, event)
                for time, event in events
                if -epsilon <= time <= remaining + epsilon
            ]
            if not valid_events:
                self.ball_x += self.ball_vx * remaining
                self.ball_y += self.ball_vy * remaining
                break

            event_time, event = min(valid_events, key=lambda item: item[0])
            event_time = max(0.0, event_time)
            self.ball_x += self.ball_vx * event_time
            self.ball_y += self.ball_vy * event_time
            remaining -= event_time

            if event == "wall":
                self.ball_y = float(np.clip(self.ball_y, min_y, max_y))
                self.ball_vy *= -1
            elif event == "player":
                if abs(self.ball_y - self.player_y) <= half_paddle + radius:
                    self.ball_x = right_plane
                    self._hits += 1
                    self._rally_count += 1
                    self._apply_paddle_bounce(self.player_y, -1, self._player_vy)
                else:
                    missed_paddles.add("player")
            elif event == "opponent":
                if abs(self.ball_y - self.opponent_y) <= half_paddle + radius:
                    self.ball_x = left_plane
                    self._rally_count += 1
                    self._apply_paddle_bounce(self.opponent_y, 1, self._opponent_vy)
                else:
                    missed_paddles.add("opponent")
            else:
                self._score_point("opponent" if event == "goal_right" else "player")
                return

    def _score_point(self, scorer: str):
        """Record a point and start the next rally."""
        if scorer == "player":
            self.player_score += 1
        else:
            self.opponent_score += 1
            self._misses += 1
        if self._rally_count > 0:
            self._rally_length_sum += self._rally_count
            self._rally_length_count += 1
        self._launch_ball()

    def _check_scoring(self):
        """Handle out-of-bounds states supplied by custom callers."""
        if self.ball_x > self.config.screen_width - self.config.offset:
            self._score_point("opponent")
        elif self.ball_x < self.config.offset:
            self._score_point("player")

    def render(self):
        """Render the environment when RGB output was requested."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None

    def _render_frame(self) -> np.ndarray:
        """Render a frame as RGB array."""
        c = self.config
        frame = np.zeros((c.screen_height, c.screen_width, 3), dtype=np.uint8)
        paddle_half = c.paddle_height // 2
        paddle_w = c.paddle_width

        p_top = int(self.player_y - paddle_half)
        p_bottom = int(self.player_y + paddle_half)
        p_left = int(self.player_x - paddle_w // 2)
        p_right = int(self.player_x + paddle_w // 2)
        frame[p_top:p_bottom, p_left:p_right] = [255, 255, 255]

        o_top = int(self.opponent_y - paddle_half)
        o_bottom = int(self.opponent_y + paddle_half)
        o_left = int(self.opponent_x - paddle_w // 2)
        o_right = int(self.opponent_x + paddle_w // 2)
        frame[o_top:o_bottom, o_left:o_right] = [255, 255, 255]

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


def register_headless_env():
    """Register the headless Pong environment with gymnasium."""
    try:
        gym.register(
            id="PongHeadless-v0",
            entry_point="pong.env.pong_headless:PongHeadlessEnv",
            max_episode_steps=10000,
        )
    except gym.error.Error:
        pass
