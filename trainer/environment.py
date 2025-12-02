"""
Environment wrapper for DQN training with Pong gym environment.
Adapted for gymnasium API and custom Pong environment.

Uses PongHeadlessEnv for fast headless training (no pygame dependency).
"""

import gymnasium as gym
import numpy as np
from pong import register_headless_env
from pong.env.pong_headless import GameConfig

# Game constants from headless config
_config = GameConfig()
SCREEN_WIDTH = _config.screen_width
SCREEN_HEIGHT = _config.screen_height
OFFSET = _config.offset


def transform_obs_for_opponent(obs):
    """
    Transform observation from player perspective to opponent perspective.

    The player is on the RIGHT side, opponent on the LEFT side.
    For the opponent's perspective, we need to mirror the X coordinates
    and swap the paddle/score positions.

    NOTE: Observations are now NORMALIZED to [0,1] or [-1,1] range.

    Args:
        obs: Normalized observation array from player perspective (9-dim)

    Returns:
        Transformed normalized observation from opponent perspective (9-dim)
    """
    ball_x_norm = obs[0]  # Normalized ball X [0, 1]
    ball_y_norm = obs[1]  # Normalized ball Y [0, 1]
    ball_vx_norm = obs[2]  # Normalized ball VX [-1, 1]
    ball_vy_norm = obs[3]  # Normalized ball VY [-1, 1]
    player_y_norm = obs[4]  # Normalized player Y [0, 1]
    opponent_y_norm = obs[5]  # Normalized opponent Y [0, 1]
    ball_dist_norm = obs[6]  # Normalized distance [0, 1]
    player_score_norm = obs[7]
    opponent_score_norm = obs[8]

    # Mirror ball X position (1 - x since normalized to [0,1])
    mirrored_ball_x = 1.0 - ball_x_norm

    # Negate ball X velocity (moving left becomes moving right from opponent view)
    mirrored_ball_vx = -ball_vx_norm

    # Calculate normalized ball distance from opponent paddle
    # Opponent paddle is at x ≈ 0 (left side) in normalized coords
    # Convert back to pixel coords for distance, then normalize
    ball_x_px = ball_x_norm * SCREEN_WIDTH
    ball_y_px = ball_y_norm * SCREEN_HEIGHT
    opponent_y_px = opponent_y_norm * SCREEN_HEIGHT
    max_distance = np.sqrt(SCREEN_WIDTH**2 + SCREEN_HEIGHT**2)
    ball_distance_to_opponent = (
        np.sqrt((ball_x_px - OFFSET) ** 2 + (ball_y_px - opponent_y_px) ** 2) / max_distance
    )

    return np.array(
        [
            mirrored_ball_x,  # Ball X (mirrored) [0, 1]
            ball_y_norm,  # Ball Y (unchanged) [0, 1]
            mirrored_ball_vx,  # Ball VX (negated) [-1, 1]
            ball_vy_norm,  # Ball VY (unchanged) [-1, 1]
            opponent_y_norm,  # Swap: opponent becomes "player" [0, 1]
            player_y_norm,  # Swap: player becomes "opponent" [0, 1]
            ball_distance_to_opponent,  # Ball distance from opponent paddle [0, 1]
            opponent_score_norm,  # Swap scores
            player_score_norm,
        ],
        dtype=np.float32,
    )


def initialize_new_game(env, agent):
    """
    Initialize a new game episode.
    Adds initial state to agent memory.

    Args:
        env: Gymnasium environment
        agent: DQN agent
    """
    obs, info = env.reset()
    # For vector observations, we just use the single state (no frame stacking)
    agent.memory.add_experience(obs, 0.0, 0, False)


def make_env(env_name, agent=None, render_mode=None, self_play=False):
    """
    Create and return a gymnasium environment.

    Uses PongHeadless-v0 for fast training (no pygame dependency).
    Falls back to Pong-v0 only when human rendering is needed.

    Args:
        env_name: Environment name (ignored, uses headless by default)
        agent: Agent instance (not used, kept for compatibility)
        render_mode: Render mode ('human', 'rgb_array', or None)
        self_play: If True, opponent is agent-controlled (not AI)

    Returns:
        Gymnasium environment instance
    """
    # Register the headless environment
    register_headless_env()

    # Use headless environment for training (much faster)
    env = gym.make("PongHeadless-v0", render_mode=render_mode, agent_controlled_opponent=self_play)
    return env


def take_step(env, agent, score, debug=False):
    """
    Take one step in the environment with self-play support.

    In self-play mode, both player and opponent use the same agent
    with noisy networks for exploration (no epsilon-greedy).

    Args:
        env: Gymnasium environment
        agent: DQN agent
        score: Current episode score
        debug: Debug mode flag

    Returns:
        Tuple of (new_score, done_flag)
    """
    # Update timesteps and save weights periodically
    agent.total_timesteps += 1
    if agent.total_timesteps % 50000 == 0:
        agent.save_weights("recent_weights.pth")
        if debug:
            print("\nWeights saved!")

    # Get current state from memory
    if len(agent.memory) > 0:
        # Get most recent observation
        idx = (agent.memory.position - 1) % agent.memory.max_len
        current_state = agent.memory.frames[idx].copy()
    else:
        # Shouldn't happen if initialize_new_game was called
        obs, _ = env.reset()
        current_state = np.array(obs, dtype=np.float32)
        agent.memory.add_experience(obs, 0.0, 0, False)

    # Get action using current state (noisy networks provide exploration)
    action = agent.get_action(current_state)

    # Self-play: get opponent action using frozen opponent network
    unwrapped_env = env.unwrapped
    if (
        hasattr(unwrapped_env, "agent_controlled_opponent")
        and unwrapped_env.agent_controlled_opponent
    ):
        # Get current observation and transform for opponent perspective
        current_obs = unwrapped_env._get_obs()
        opponent_obs = transform_obs_for_opponent(current_obs)

        # Get opponent action using frozen network (stable training signal)
        # The opponent network is updated periodically, providing consistent challenge
        opponent_action = agent.get_opponent_action(opponent_obs)

        unwrapped_env.set_opponent_action(opponent_action)

    # Take step in environment
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Add experience to memory
    agent.memory.add_experience(obs, reward, action, done)

    # Render if needed
    if debug or env.render_mode is not None:
        env.render()

    # Learn if we have enough experiences (every N steps for performance)
    if (
        len(agent.memory) > agent.starting_mem_len
        and agent.total_timesteps % agent.learn_every == 0
    ):
        agent.learn(debug)

    new_score = score + reward
    return new_score, done


def play_episode(env, agent, debug=False):
    """
    Play one complete episode.

    Args:
        env: Gymnasium environment
        agent: DQN agent
        debug: Debug mode flag

    Returns:
        Final episode score
    """
    initialize_new_game(env, agent)
    done = False
    score = 0.0

    while not done:
        score, done = take_step(env, agent, score, debug)

    return score


def play_episode_with_info(env, agent, debug=False):
    """
    Play one complete episode and return info for curriculum tracking.

    Args:
        env: Gymnasium environment
        agent: DQN agent
        debug: Debug mode flag

    Returns:
        Tuple of (final_score, final_info_dict)
    """
    initialize_new_game(env, agent)
    done = False
    score = 0.0
    info = {}

    while not done:
        score, done = take_step(env, agent, score, debug)
    
    # Get final info from environment for curriculum metrics
    unwrapped_env = env.unwrapped
    if hasattr(unwrapped_env, '_get_info'):
        info = unwrapped_env._get_info()
    
    return score, info
