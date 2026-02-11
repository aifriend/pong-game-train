"""
Evaluation Suite for Pong AI Agent.

Evaluates trained PPO agents against multiple opponent types and reports
performance metrics to validate master-level achievement.

Master-Level Benchmarks:
- slow_ai: >= 90% win rate
- beginner_ai: >= 75% win rate
- normal_ai: >= 55% win rate
- reactive_ai: >= 50% win rate

Usage:
    python scripts/evaluate_agent.py --weights models/ppo_final.zip --episodes 20
"""

import argparse
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pong.env.pong_headless import PongHeadlessEnv, OpponentType
from stable_baselines3 import PPO


# Master-level benchmarks
BENCHMARKS = {
    OpponentType.SLOW_AI: 0.90,      # Should easily beat
    OpponentType.BEGINNER_AI: 0.75,  # Phase 1 validation
    OpponentType.NORMAL_AI: 0.55,    # Phase 2 validation
    OpponentType.REACTIVE_AI: 0.50,  # Phase 3 validation
}


def load_model(weights_path: str):
    """
    Load Stable-Baselines3 model from .zip file.
    
    Args:
        weights_path: Path to model file (.zip)
        
    Returns:
        Loaded SB3 model
    """
    path = Path(weights_path)
    
    # Add .zip if not present
    if path.suffix != ".zip":
        path = Path(str(path) + ".zip")
    
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    
    model = PPO.load(str(path))
    print(f"✅ Loaded PPO model from {path}")
    return model


def evaluate_against_opponent(
    model,
    opponent_type: OpponentType,
    n_episodes: int = 20,
    ball_speed: float = 1.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate model against a specific opponent type.
    
    Args:
        model: Trained SB3 model
        opponent_type: Type of opponent to play against
        n_episodes: Number of evaluation episodes
        ball_speed: Ball speed multiplier
        verbose: Print progress
        
    Returns:
        Dictionary with evaluation results
    """
    env = PongHeadlessEnv(
        ball_speed_multiplier=ball_speed,
        opponent_type=opponent_type,
        agent_controlled_opponent=False,
    )
    
    wins = 0
    losses = 0
    total_score_diff = 0
    total_rallies = 0
    rally_counts = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        # Track results
        player_score = info.get("player_score", 0)
        opponent_score = info.get("opponent_score", 0)
        
        if player_score > opponent_score:
            wins += 1
        else:
            losses += 1
        
        total_score_diff += player_score - opponent_score
        avg_rally = info.get("avg_rally", 0)
        total_rallies += avg_rally
        rally_counts.append(avg_rally)
        
        if verbose:
            result = "WIN" if player_score > opponent_score else "LOSS"
            print(f"  Episode {ep+1}/{n_episodes}: {result} ({player_score}-{opponent_score})")
    
    env.close()
    
    win_rate = wins / n_episodes
    avg_score_diff = total_score_diff / n_episodes
    avg_rally = total_rallies / n_episodes
    rally_std = np.std(rally_counts) if rally_counts else 0
    
    return {
        "opponent": opponent_type.value,
        "n_episodes": n_episodes,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_score_diff": avg_score_diff,
        "avg_rally": avg_rally,
        "rally_std": rally_std,
    }


def run_full_evaluation(
    model,
    n_episodes: int = 20,
    verbose: bool = True,
) -> Tuple[Dict[str, Dict[str, Any]], bool]:
    """
    Run full evaluation against all opponent types.
    
    Args:
        model: Trained SB3 model
        n_episodes: Episodes per opponent
        verbose: Print progress
        
    Returns:
        Tuple of (results_dict, passed_all_benchmarks)
    """
    results = {}
    all_passed = True
    
    opponents = [
        (OpponentType.SLOW_AI, "Slow AI"),
        (OpponentType.BEGINNER_AI, "Beginner AI"),
        (OpponentType.NORMAL_AI, "Normal AI"),
        (OpponentType.REACTIVE_AI, "Reactive AI"),
    ]
    
    print("\n" + "=" * 60)
    print("🎯 PONG AI EVALUATION SUITE")
    print("=" * 60)
    
    for opponent_type, name in opponents:
        print(f"\n📊 Evaluating vs {name}...")
        
        result = evaluate_against_opponent(
            model,
            opponent_type,
            n_episodes,
            verbose=verbose,
        )
        results[opponent_type.value] = result
        
        # Check benchmark
        benchmark = BENCHMARKS.get(opponent_type, 0.5)
        passed = result["win_rate"] >= benchmark
        
        if not passed:
            all_passed = False
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"\n  Results vs {name}:")
        print(f"    Win Rate: {result['win_rate']*100:.1f}% (benchmark: {benchmark*100:.0f}%) {status}")
        print(f"    Record: {result['wins']}-{result['losses']}")
        print(f"    Avg Score Diff: {result['avg_score_diff']:+.1f}")
        print(f"    Avg Rally: {result['avg_rally']:.1f} (σ={result['rally_std']:.2f})")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 EVALUATION SUMMARY")
    print("=" * 60)
    
    for opponent_type, name in opponents:
        result = results[opponent_type.value]
        benchmark = BENCHMARKS.get(opponent_type, 0.5)
        passed = result["win_rate"] >= benchmark
        status = "✅" if passed else "❌"
        print(f"  {status} {name}: {result['win_rate']*100:.1f}% (need {benchmark*100:.0f}%)")
    
    print("\n" + "-" * 60)
    if all_passed:
        print("🏆 MASTER LEVEL ACHIEVED! All benchmarks passed!")
    else:
        print("⚠️  Some benchmarks not met. Continue training.")
    print("-" * 60)
    
    return results, all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Pong AI agent (Stable-Baselines3 models)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/ppo_final.zip",
        help="Path to model file (.zip)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of episodes per opponent",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-episode output",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default=None,
        choices=["slow_ai", "beginner_ai", "normal_ai", "reactive_ai"],
        help="Evaluate against specific opponent only",
    )
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.weights}...")
    
    try:
        model = load_model(args.weights)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1
    except ValueError as e:
        print(f"❌ {e}")
        return 1
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1
    
    # Single opponent evaluation
    if args.opponent:
        opponent_mapping = {
            "slow_ai": OpponentType.SLOW_AI,
            "beginner_ai": OpponentType.BEGINNER_AI,
            "normal_ai": OpponentType.NORMAL_AI,
            "reactive_ai": OpponentType.REACTIVE_AI,
        }
        opponent_type = opponent_mapping[args.opponent]
        
        print(f"\n📊 Evaluating vs {args.opponent}...")
        result = evaluate_against_opponent(
            model,
            opponent_type,
            n_episodes=args.episodes,
            verbose=not args.quiet,
        )
        
        benchmark = BENCHMARKS.get(opponent_type, 0.5)
        passed = result["win_rate"] >= benchmark
        status = "✅ PASS" if passed else "❌ FAIL"
        
        print(f"\n  Win Rate: {result['win_rate']*100:.1f}% (benchmark: {benchmark*100:.0f}%) {status}")
        print(f"  Record: {result['wins']}-{result['losses']}")
        
        return 0 if passed else 1
    
    # Full evaluation
    results, passed = run_full_evaluation(
        model,
        n_episodes=args.episodes,
        verbose=not args.quiet,
    )
    
    # Return exit code based on benchmark results
    return 0 if passed else 1


if __name__ == "__main__":
    exit(main() or 0)
