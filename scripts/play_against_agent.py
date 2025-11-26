"""
Play against a trained agent.

You control the player paddle (right side) with UP/DOWN arrows.
The agent controls the opponent paddle (left side).

Usage:
    python scripts/play_against_agent.py <checkpoint_path>

Example:
    python scripts/play_against_agent.py checkpoints/checkpoint_episode_2000.pth
"""

from pong.constants import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    SCREEN_SIZE,
    OFFSET,
    FPS,
    AUDIO_FREQUENCY,
    AUDIO_SIZE,
    AUDIO_CHANNELS,
    AUDIO_BUFFER,
)
from pong.game.game_object import GameObject
from trainer.environment import transform_obs_for_opponent
from trainer import Agent
import numpy as np
import pygame
import sys
import os
from pathlib import Path

# Add project root to path so imports work when running directly
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# Agent configuration (must match training)
POSSIBLE_ACTIONS = [0, 1, 2]  # 0: stay, 1: up, 2: down
OBSERVATION_DIM = 9
MAX_SCORE = 11


def get_observation(game_manager, player, opponent):
    """Get NORMALIZED observation from player perspective (matches training)."""
    from pong.game.ball import Ball

    ball = game_manager.ball_group.sprite
    ball_x = float(ball.rect.centerx)
    ball_y = float(ball.rect.centery)
    ball_vx = float(ball.speed_x)
    ball_vy = float(ball.speed_y)
    player_y = float(player.rect.centery)
    opponent_y = float(opponent.rect.centery)

    max_distance = np.sqrt(SCREEN_WIDTH**2 + SCREEN_HEIGHT**2)
    ball_distance = np.sqrt((ball_x - player.rect.centerx) ** 2 + (ball_y - player_y) ** 2)

    player_score = float(game_manager.player_score)
    opponent_score = float(game_manager.opponent_score)

    # Return NORMALIZED observation (same as training environment)
    return np.array(
        [
            ball_x / SCREEN_WIDTH,  # [0, 1]
            ball_y / SCREEN_HEIGHT,  # [0, 1]
            ball_vx / Ball.HIGH_SPEED,  # [-1, 1]
            ball_vy / Ball.HIGH_SPEED,  # [-1, 1]
            player_y / SCREEN_HEIGHT,  # [0, 1]
            opponent_y / SCREEN_HEIGHT,  # [0, 1]
            ball_distance / max_distance,  # [0, 1]
            player_score / MAX_SCORE,  # [0, 1]
            opponent_score / MAX_SCORE,  # [0, 1]
        ],
        dtype=np.float32,
    )


def main():
    """Main game loop where human plays against trained agent."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/play_against_agent.py <checkpoint_path>")
        print(
            "Example: python scripts/play_against_agent.py checkpoints/checkpoint_episode_2000.pth"
        )
        sys.exit(1)

    checkpoint_path = sys.argv[1]

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)

    print(f"Loading agent from: {checkpoint_path}")

    # Initialize agent (noisy networks handle exploration, disabled during eval)
    agent = Agent(
        possible_actions=POSSIBLE_ACTIONS,
        starting_mem_len=10000,
        max_mem_len=100000,
        learn_rate=0.001,
        observation_dim=OBSERVATION_DIM,
        learn_every=8,
        batch_size=128,
        target_update_freq=5000,
    )

    # Load weights
    agent.load_weights(checkpoint_path)
    print("Agent loaded successfully!")

    # Initialize pygame
    pygame.mixer.pre_init(AUDIO_FREQUENCY, AUDIO_SIZE, AUDIO_CHANNELS, AUDIO_BUFFER)
    pygame.init()
    clock = pygame.time.Clock()
    pygame.display.set_caption("Pong - Play Against Agent")

    # Create game objects (opponent with manual_control=True for agent control)
    game_manager, player, opponent = GameObject.get_game_object(
        screen_size=SCREEN_SIZE,
        offset=OFFSET,
        screen_width=SCREEN_WIDTH,
        screen_height=SCREEN_HEIGHT,
        opponent_manual_control=True,  # Agent controls opponent
    )

    print("\n" + "=" * 50)
    print("🎮 PONG - Human vs Agent")
    print("=" * 50)
    print("\nControls:")
    print("   UP arrow    : Move paddle up")
    print("   DOWN arrow  : Move paddle down")
    print("   ESC         : Quit")
    print("\nYou are the PLAYER (right paddle)")
    print("The agent is the OPPONENT (left paddle)")
    print(f"First to {MAX_SCORE} wins!")
    print("=" * 50 + "\n")

    running = True
    games_played = 0
    human_wins = 0
    agent_wins = 0

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_UP:
                    player.movement -= player.speed
                elif event.key == pygame.K_DOWN:
                    player.movement += player.speed
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    player.movement += player.speed
                elif event.key == pygame.K_DOWN:
                    player.movement -= player.speed

        # Get observation for agent (opponent perspective)
        obs = get_observation(game_manager, player, opponent)
        opponent_obs = transform_obs_for_opponent(obs)

        # Get agent action for opponent paddle (greedy, no exploration)
        opponent_action = agent.get_action(opponent_obs, use_noise=False)

        # Apply action to opponent paddle
        if opponent_action == 1:  # Move up
            opponent.movement = -opponent.speed
        elif opponent_action == 2:  # Move down
            opponent.movement = opponent.speed
        else:  # Stay
            opponent.movement = 0

        # Update game state
        game_manager.paddle_group.update(game_manager.ball_group)
        game_manager.ball_group.update(game_manager)
        game_manager.check_score()

        # Draw game
        game_manager.background()
        game_manager.run_game()

        # Display
        pygame.display.update()
        clock.tick(FPS)

        # Check if game ended
        if game_manager.player_score >= MAX_SCORE or game_manager.opponent_score >= MAX_SCORE:
            games_played += 1

            if game_manager.player_score > game_manager.opponent_score:
                human_wins += 1
                result = "🏆 YOU WIN!"
            else:
                agent_wins += 1
                result = "🤖 AGENT WINS!"

            print(f"\n{result}")
            print(
                f"Final Score: You {game_manager.player_score} - {game_manager.opponent_score} Agent"
            )
            print(f"Overall: You {human_wins} - {agent_wins} Agent ({games_played} games)")
            print("\nStarting new game...")

            # Reset game
            game_manager, player, opponent = GameObject.get_game_object(
                screen_size=SCREEN_SIZE,
                offset=OFFSET,
                screen_width=SCREEN_WIDTH,
                screen_height=SCREEN_HEIGHT,
                opponent_manual_control=True,
            )

    # Final stats
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"Games played: {games_played}")
    print(f"Human wins: {human_wins}")
    print(f"Agent wins: {agent_wins}")
    if games_played > 0:
        print(f"Human win rate: {100 * human_wins / games_played:.1f}%")
    print("=" * 50)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
