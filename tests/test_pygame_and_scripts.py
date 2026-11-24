"""Pygame smoke tests (SDL dummy drivers) and evaluate_agent script parsing tests."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest

from pong.constants import OFFSET, SCREEN_HEIGHT, SCREEN_SIZE, SCREEN_WIDTH
from pong.env.pong_headless import OpponentType
from pong.env.wrappers import make_ppo_env
from pong.game.game_object import GameObject
from scripts import evaluate_agent


@pytest.fixture
def pygame_display():
    pygame.init()
    pygame.display.set_mode(SCREEN_SIZE)
    yield
    pygame.quit()


def test_game_objects_run_frames_with_dummy_video(pygame_display):
    game_manager, player, opponent = GameObject.get_game_object(
        screen_size=SCREEN_SIZE,
        offset=OFFSET,
        screen_width=SCREEN_WIDTH,
        screen_height=SCREEN_HEIGHT,
    )
    for _ in range(30):
        game_manager.background()
        game_manager.run_game()
        pygame.display.update()
    assert player is not None and opponent is not None


def test_make_ppo_env_accepts_enum_and_string():
    from_str = make_ppo_env(opponent_type="medium_ai")
    from_enum = make_ppo_env(opponent_type=OpponentType.MEDIUM_AI)
    assert from_str.unwrapped.opponent_type == OpponentType.MEDIUM_AI
    assert from_enum.unwrapped.opponent_type == OpponentType.MEDIUM_AI


def test_medium_ai_has_benchmark():
    assert OpponentType.MEDIUM_AI in evaluate_agent.BENCHMARKS


def test_evaluate_agent_cli_accepts_medium_ai(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["evaluate_agent", "--opponent", "medium_ai"])
    monkeypatch.setattr(
        evaluate_agent,
        "load_model",
        lambda path: (_ for _ in ()).throw(FileNotFoundError("no weights")),
    )
    assert evaluate_agent.main() == 1
    assert "no weights" in capsys.readouterr().out


def test_evaluate_agent_cli_rejects_unknown_opponent(monkeypatch):
    monkeypatch.setattr("sys.argv", ["evaluate_agent", "--opponent", "bogus_ai"])
    with pytest.raises(SystemExit) as exc:
        evaluate_agent.main()
    assert exc.value.code == 2
