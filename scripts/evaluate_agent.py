"""
Evaluation Suite for Pong AI Agent.

Tests the trained agent against multiple opponent types and reports
performance metrics to validate master-level achievement.

Master-Level Benchmarks:
- slow_ai: >= 90% win rate
- beginner_ai: >= 75% win rate
- normal_ai: >= 55% win rate
- reactive_ai: >= 50% win rate

Usage:
    python scripts/evaluate_agent.py --weights final_weights.pth --episodes 20
"""

import argparse
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pong.env.pong_headless import PongHeadlessEnv, OpponentType
from trainer import Agent


# Master-level benchmarks
BENCHMARKS = {
    OpponentType.SLOW_AI: 0.90,      # Should easily beat
    OpponentType.BEGINNER_AI: 0.75,  # Phase 3 validation
    OpponentType.NORMAL_AI: 0.55,    # Phase 4 validation
    OpponentType.REACTIVE_AI: 0.50,  # Phase 5 validation
}


def evaluate_against_opponent(
    agent: Agent,
    opponent_type: OpponentType,
    n_episodes: int = 20,
    ball_speed: float = 1.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate agent against a specific opponent type.
    
    Args:
        agent: Trained DQN agent
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
        episode_score = 0
        
        while not done:
            # Get action from agent (no noise during evaluation)
            action = agent.get_action(obs, use_noise=False)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_score += reward
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
    agent: Agent,
    n_episodes: int = 20,
    verbose: bool = True,
) -> Tuple[Dict[str, Dict[str, Any]], bool]:
    """
    Run full evaluation against all opponent types.
    
    Args:
        agent: Trained DQN agent
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
            agent,
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
    parser = argparse.ArgumentParser(description="Evaluate Pong AI agent")
    parser.add_argument(
        "--weights",
        type=str,
        default="final_weights.pth",
        help="Path to weights file",
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
    args = parser.parse_args()
    
    # Load agent
    print(f"Loading agent from {args.weights}...")
    
    agent = Agent(
        possible_actions=[0, 1, 2],
        starting_mem_len=1000,
        max_mem_len=10000,
        learn_rate=0.001,
        observation_dim=9,
    )
    
    try:
        agent.load_weights(args.weights)
        print("✅ Weights loaded successfully")
    except FileNotFoundError:
        print(f"❌ Weights file not found: {args.weights}")
        return
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return
    
    # Run evaluation
    results, passed = run_full_evaluation(
        agent,
        n_episodes=args.episodes,
        verbose=not args.quiet,
    )
    
    # Return exit code based on benchmark results
    return 0 if passed else 1


if __name__ == "__main__":
    exit(main() or 0)

