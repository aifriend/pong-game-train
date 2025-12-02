"""Constants and configuration for the Pong game."""

from pathlib import Path

# Screen configuration
SCREEN_WIDTH = 960
SCREEN_HEIGHT = 720
SCREEN_SIZE = (SCREEN_WIDTH, SCREEN_HEIGHT)
OFFSET = 20
FPS = 120

# Colors
BG_COLOR = "#000000"
ACCENT_COLOR = (151, 151, 151)

# Ball configuration
BALL_LOW_SPEED = 2
BALL_HIGH_SPEED = 4
BALL_SMALL_PAD = 1
BALL_LARGE_PAD = 2
BALL_ANGLE_20 = 20
BALL_ANGLE_40 = 40

# Opponent AI configuration
OPPONENT_MIN_SPEED = 2
OPPONENT_MAX_SPEED = 5
OPPONENT_RANDOM_RANGE = 30

# Score display
SCORE_RESOLUTION = 15
OPPONENT_SCORE_X_MULTIPLIER = 6
PLAYER_SCORE_X_MULTIPLIER = 9
SCORE_Y_MULTIPLIER = 1.7

# Resource paths
BASE_DIR = Path(__file__).parent.parent  # Go up from pong/ to project root
RESOURCES_DIR = BASE_DIR / "resources"
BALL_IMAGE = RESOURCES_DIR / "ball.png"
PADDLE_IMAGE = RESOURCES_DIR / "paddle.png"
PONG_SOUND = RESOURCES_DIR / "pong.ogg"
SCORE_SOUND = RESOURCES_DIR / "score.ogg"
FONT_PATH = RESOURCES_DIR / "bit5x3.ttf"
FALLBACK_FONT = "freesansbold.ttf"

# Font sizes
FONT_SIZE_BASIC = 32
FONT_SIZE_SCORE = 120

# Collision detection
COLLISION_THRESHOLD = 10

# Audio configuration
AUDIO_FREQUENCY = 44100
AUDIO_SIZE = -16
AUDIO_CHANNELS = 2
AUDIO_BUFFER = 512

# Player movement configuration
PLAYER_ACCELERATION = 0.2
PLAYER_DECELERATION = 0.7
PLAYER_MAX_SPEED = 8.0
