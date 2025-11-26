#!/usr/bin/env python3
"""
Comprehensive test to prove the Pong project is working correctly.
Tests all major components without hanging.
"""
import os
import sys
import traceback

# Set environment variables to prevent pygame from hanging
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


def test_imports():
    """Test that all packages can be imported."""
    print("=" * 60)
    print("TEST 1: Package Imports")
    print("=" * 60)

    results = []

    # Test basic dependencies
    try:
        import numpy as np

        print(f"✓ numpy {np.__version__}")
        results.append(True)
    except Exception as e:
        print(f"✗ numpy: {e}")
        results.append(False)

    try:
        import gymnasium as gym

        print(f"✓ gymnasium {gym.__version__}")
        results.append(True)
    except Exception as e:
        print(f"✗ gymnasium: {e}")
        results.append(False)

    try:
        import torch

        print(f"✓ torch {torch.__version__}")
        results.append(True)
    except Exception as e:
        print(f"✗ torch: {e}")
        results.append(False)

    # Test pong package
    try:
        from pong import PongEnv, register_pong_env

        print("✓ pong package imported")
        results.append(True)
    except Exception as e:
        print(f"✗ pong package: {e}")
        traceback.print_exc()
        results.append(False)

    # Test trainer package
    try:
        from trainer import Agent, Memory

        print("✓ trainer package imported")
        results.append(True)
    except Exception as e:
        print(f"✗ trainer package: {e}")
        traceback.print_exc()
        results.append(False)

    return all(results)


def test_environment_registration():
    """Test environment registration."""
    print("\n" + "=" * 60)
    print("TEST 2: Environment Registration")
    print("=" * 60)

    try:
        from pong import register_pong_env
        import gymnasium as gym

        register_pong_env()
        print("✓ Environment registered")

        # Try to make the environment
        env = gym.make("Pong-v0", render_mode=None)
        print("✓ Environment created successfully")

        env.close()
        print("✓ Environment closed successfully")
        return True
    except Exception as e:
        print(f"✗ Environment registration failed: {e}")
        traceback.print_exc()
        return False


def test_environment_functionality():
    """Test basic environment functionality."""
    print("\n" + "=" * 60)
    print("TEST 3: Environment Functionality")
    print("=" * 60)

    try:
        import gymnasium as gym
        from pong import register_pong_env

        register_pong_env()
        env = gym.make("Pong-v0", render_mode=None, max_score=3, max_steps=100)

        # Test reset
        obs, info = env.reset()
        print(f"✓ Reset successful - observation shape: {obs.shape}")
        assert obs.shape == (9,), f"Expected shape (9,), got {obs.shape}"
        print(f"✓ Observation space correct: {obs.shape}")

        # Test action space
        assert env.action_space.n == 3, "Action space should have 3 actions"
        print(f"✓ Action space correct: {env.action_space.n} actions")

        # Test a few steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (9,), f"Observation shape incorrect at step {i}"
            assert isinstance(reward, (int, float)), f"Reward type incorrect at step {i}"
            assert isinstance(terminated, bool), f"Terminated type incorrect at step {i}"
            assert isinstance(truncated, bool), f"Truncated type incorrect at step {i}"

            if terminated or truncated:
                obs, info = env.reset()

        print("✓ Multiple steps executed successfully")
        print(f"✓ Final scores: Player {info['player_score']}, Opponent {info['opponent_score']}")

        env.close()
        return True
    except Exception as e:
        print(f"✗ Environment functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_agent_creation():
    """Test agent can be created."""
    print("\n" + "=" * 60)
    print("TEST 4: Agent Creation")
    print("=" * 60)

    try:
        from trainer import Agent

        agent = Agent(
            possible_actions=[0, 1, 2],
            starting_mem_len=100,
            max_mem_len=1000,
            learn_rate=0.001,
            observation_dim=9,
            learn_every=8,
            batch_size=128,
            target_update_freq=5000,
        )
        print("✓ Agent created successfully")
        print(f"✓ Agent device: {agent.device}")
        print(f"✓ Learn every: {agent.learn_every} steps")
        print(f"✓ Batch size: {agent.batch_size}")
        print(f"✓ Memory size: {len(agent.memory.frames)}")
        return True
    except Exception as e:
        print(f"✗ Agent creation failed: {e}")
        traceback.print_exc()
        return False


def test_game_objects():
    """Test game object creation."""
    print("\n" + "=" * 60)
    print("TEST 5: Game Object Creation")
    print("=" * 60)

    try:
        import pygame

        # Initialize pygame for font system
        pygame.init()

        from pong.game.game_object import GameObject
        from pong.constants import SCREEN_SIZE, OFFSET, SCREEN_WIDTH, SCREEN_HEIGHT

        game_manager, player, opponent = GameObject.get_game_object(
            screen_size=SCREEN_SIZE,
            offset=OFFSET,
            screen_width=SCREEN_WIDTH,
            screen_height=SCREEN_HEIGHT,
        )

        print("✓ Game objects created successfully")
        print(f"✓ Player position: {player.rect.center}")
        print(f"✓ Opponent position: {opponent.rect.center}")
        print(f"✓ Ball position: {game_manager.ball_group.sprite.rect.center}")

        pygame.quit()
        return True
    except Exception as e:
        print(f"✗ Game object creation failed: {e}")
        traceback.print_exc()
        return False


def test_constants():
    """Test constants are properly defined."""
    print("\n" + "=" * 60)
    print("TEST 6: Constants Validation")
    print("=" * 60)

    try:
        from pong.constants import (
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            SCREEN_SIZE,
            OFFSET,
            FPS,
            BALL_LOW_SPEED,
            BALL_HIGH_SPEED,
            OPPONENT_MIN_SPEED,
            OPPONENT_MAX_SPEED,
            OPPONENT_RANDOM_RANGE,
        )

        assert SCREEN_WIDTH > 0, "SCREEN_WIDTH must be positive"
        assert SCREEN_HEIGHT > 0, "SCREEN_HEIGHT must be positive"
        assert len(SCREEN_SIZE) == 2, "SCREEN_SIZE must be a tuple of 2"
        assert OFFSET > 0, "OFFSET must be positive"
        assert FPS > 0, "FPS must be positive"

        print("✓ All constants properly defined")
        print(f"  Screen: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
        print(f"  FPS: {FPS}")
        print(f"  Ball speeds: {BALL_LOW_SPEED} - {BALL_HIGH_SPEED}")
        print(f"  Opponent speed range: {OPPONENT_MIN_SPEED} - {OPPONENT_MAX_SPEED}")
        return True
    except Exception as e:
        print(f"✗ Constants validation failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PONG PROJECT - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print("\nTesting all components to prove everything works...\n")

    tests = [
        ("Imports", test_imports),
        ("Environment Registration", test_environment_registration),
        ("Environment Functionality", test_environment_functionality),
        ("Agent Creation", test_agent_creation),
        ("Game Objects", test_game_objects),
        ("Constants", test_constants),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print("\n" + "=" * 60)
    if passed == total:
        print(f"🎉 SUCCESS: All {total} tests passed!")
        print("✅ The project is working correctly!")
        return 0
    else:
        print(f"⚠️  {passed}/{total} tests passed")
        print("Some tests failed - see details above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
