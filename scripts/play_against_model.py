"""
Play Against Trained PPO Model.

Interactive pygame script: human (right paddle) vs PPO agent (left paddle).
Uses the same headless env physics the model was trained on.

Controls:
    UP / DOWN arrows  - Move paddle
    ESC / Close window - Quit

Usage:
    PYTHONPATH=. python scripts/play_against_model.py --weights models_v2/ppo_final.zip
    PYTHONPATH=. python scripts/play_against_model.py --weights models_v2/ppo_final.zip --speed 0.8
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pygame
from stable_baselines3 import PPO

from pong.env.pong_headless import PongHeadlessEnv, OpponentType, GameConfig


def build_mirrored_obs(env: PongHeadlessEnv) -> np.ndarray:
    """
    Build a mirrored observation so the PPO model (trained as right paddle)
    can control the left paddle.

    Flips horizontal positions/velocities and swaps player <-> opponent.
    """
    c = env.config
    max_speed = c.base_ball_speed * 2

    ball_x_norm = 1.0 - (env.ball_x / c.screen_width)  # flip horizontal
    ball_y_norm = env.ball_y / c.screen_height
    ball_vx_norm = np.clip(-env.ball_vx / max_speed, -1, 1)  # flip vx
    ball_vy_norm = np.clip(env.ball_vy / max_speed, -1, 1)

    # Swap: model sees opponent_y as "its paddle", player_y as "the other"
    model_paddle_norm = env.opponent_y / c.screen_height
    other_paddle_norm = env.player_y / c.screen_height

    # Distance (symmetric, same either way)
    dx = env.ball_x - env.opponent_x
    dy = env.ball_y - env.opponent_y
    max_dist = np.sqrt(c.screen_width**2 + c.screen_height**2)
    dist_norm = np.clip(np.sqrt(dx**2 + dy**2) / max_dist, 0, 1)

    # Swap scores: model's score = opponent_score, other = player_score
    model_score_norm = env.opponent_score / c.max_score
    other_score_norm = env.player_score / c.max_score

    return np.array(
        [
            ball_x_norm,
            ball_y_norm,
            ball_vx_norm,
            ball_vy_norm,
            model_paddle_norm,
            other_paddle_norm,
            dist_norm,
            model_score_norm,
            other_score_norm,
        ],
        dtype=np.float32,
    )


def apply_model_action(env: PongHeadlessEnv, action: int):
    """Move opponent paddle based on model's action."""
    c = env.config
    if action == 1:  # up
        env.opponent_y -= c.base_paddle_speed
    elif action == 2:  # down
        env.opponent_y += c.base_paddle_speed

    half = c.paddle_height / 2
    env.opponent_y = np.clip(
        env.opponent_y,
        c.offset + half,
        c.screen_height - c.offset - half,
    )


def main():
    parser = argparse.ArgumentParser(description="Play Pong against trained PPO model")
    parser.add_argument(
        "--weights",
        type=str,
        default="models_v2/ppo_final.zip",
        help="Path to PPO model .zip file",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Ball speed multiplier (default: 1.0)"
    )
    args = parser.parse_args()

    # --- Load model ---
    weights = Path(args.weights)
    if not weights.exists() and not weights.with_suffix(".zip").exists():
        print(f"Model not found: {weights}")
        sys.exit(1)
    model = PPO.load(str(weights))
    print(f"Loaded model: {weights}")

    # --- Create environment ---
    # Use SLOW_AI as dummy opponent (we'll override its movement)
    config = GameConfig()
    env = PongHeadlessEnv(
        render_mode="rgb_array",
        ball_speed_multiplier=args.speed,
        opponent_type=OpponentType.SLOW_AI,
        config=config,
    )
    # Override AI movement so it does nothing (model controls opponent)
    env._move_opponent = lambda: None

    # --- Pygame setup ---
    pygame.init()
    screen = pygame.display.set_mode((config.screen_width, config.screen_height))
    pygame.display.set_caption("Pong — YOU (right) vs PPO AI (left)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 48, bold=True)
    small_font = pygame.font.SysFont("monospace", 24)

    obs, _ = env.reset()
    running = True
    game_over = False
    game_over_timer = 0

    print("\n🏓 PONG — Use UP/DOWN arrows. ESC to quit.\n")

    while running:
        # --- Events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        if not running:
            break

        # --- Handle game-over pause ---
        if game_over:
            pygame.time.wait(50)
            game_over_timer -= 50
            if game_over_timer <= 0:
                obs, _ = env.reset()
                game_over = False
            continue

        # --- Human input (right paddle) ---
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            human_action = 1
        elif keys[pygame.K_DOWN]:
            human_action = 2
        else:
            human_action = 0

        # --- PPO model action (left paddle) ---
        mirrored_obs = build_mirrored_obs(env)
        model_action, _ = model.predict(mirrored_obs, deterministic=True)
        apply_model_action(env, int(model_action))

        # --- Step environment ---
        obs, reward, terminated, truncated, info = env.step(human_action)

        # --- Render ---
        frame = env._render_frame()  # (H, W, 3) uint8
        surface = pygame.surfarray.make_surface(
            np.transpose(frame, (1, 0, 2))  # pygame wants (W, H, 3)
        )
        screen.blit(surface, (0, 0))

        # Score overlay
        score_text = font.render(
            f"{env.opponent_score}   {env.player_score}",
            True,
            (255, 255, 255),
        )
        screen.blit(
            score_text, (config.screen_width // 2 - score_text.get_width() // 2, 20)
        )

        # Labels
        ai_label = small_font.render("AI", True, (180, 180, 180))
        you_label = small_font.render("YOU", True, (180, 180, 180))
        screen.blit(
            ai_label, (config.screen_width // 4 - ai_label.get_width() // 2, 70)
        )
        screen.blit(
            you_label, (3 * config.screen_width // 4 - you_label.get_width() // 2, 70)
        )

        # Center dotted line
        for y in range(0, config.screen_height, 20):
            pygame.draw.rect(
                screen, (80, 80, 80), (config.screen_width // 2 - 1, y, 2, 10)
            )

        # --- Game over detection ---
        if terminated:
            if env.player_score > env.opponent_score:
                result = "YOU WIN!"
                color = (0, 255, 100)
            else:
                result = "AI WINS"
                color = (255, 80, 80)
            result_text = font.render(result, True, color)
            screen.blit(
                result_text,
                (
                    config.screen_width // 2 - result_text.get_width() // 2,
                    config.screen_height // 2 - result_text.get_height() // 2,
                ),
            )
            game_over = True
            game_over_timer = 2000  # 2 seconds pause

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    print("Thanks for playing!")


if __name__ == "__main__":
    main()
