#!/usr/bin/env python3
"""
RL coach CLI: run NeoHorse-Jev-4B against Pong training metrics and turn its
decisions into bounded config proposals.

Commands:
    collect              Print the metrics summary the coach reasons over.
    propose              Run one coaching session, write a proposal JSON.
    apply <proposal>     Merge an approved proposal into coach/overrides.json.
    status               Show the active overrides.

Examples:
    python scripts/coach.py collect
    python scripts/coach.py --phase 4 propose --backend neohorse --model-dir models_neohorse/NeoHorse-Jev-4B
    python scripts/coach.py apply coach/proposals/proposal_*.json
    python scripts/train_ppo_curriculum.py --phase 3 --overrides coach/overrides.json
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pong.coach.apply import (
    apply_proposal,
    format_overrides,
)
from pong.coach.metrics import build_state_text, collect_metrics
from pong.coach.propose import format_proposal, run_proposal

DEFAULT_MODEL_DIR = project_root / "models_neohorse" / "NeoHorse-Jev-4B"
OVERRIDES_PATH = project_root / "coach" / "overrides.json"
PROPOSALS_DIR = project_root / "coach" / "proposals"


def _parse_obs_features(raw: Optional[str]):
    if not raw:
        return None
    return [f.strip() for f in raw.split(",") if f.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=str,
        default=str(project_root / "models" / "ppo_final"),
        help="Model to evaluate (default: models/ppo_final)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Evaluation episodes per opponent (default: 20)",
    )
    parser.add_argument(
        "--obs-features",
        type=str,
        default=None,
        help="Comma-separated observation features the model was trained with "
        "(needed when evaluating a model trained with coach-enabled features)",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=None,
        help="Phase to inspect/coach (default: latest phase found in tensorboard logs)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("collect", help="Print the metrics summary")

    propose_parser = subparsers.add_parser("propose", help="Run a coaching session")
    propose_parser.add_argument(
        "--backend",
        choices=["neohorse", "mock"],
        default="mock",
        help="Decision backend (default: mock)",
    )
    propose_parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help="NeoHorse-Jev-4B bundle directory (neohorse backend)",
    )
    propose_parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip the fresh model evaluation (tensorboard + config only)",
    )

    apply_parser = subparsers.add_parser(
        "apply", help="Merge an approved proposal into coach/overrides.json"
    )
    apply_parser.add_argument("proposal", type=str)

    subparsers.add_parser("status", help="Show active overrides")

    args = parser.parse_args()
    obs_features = _parse_obs_features(args.obs_features)

    if args.command == "collect":
        metrics = collect_metrics(
            project_root,
            model_path=Path(args.model),
            n_episodes=args.episodes,
            obs_features=obs_features,
        )
        phase = args.phase if args.phase else metrics.get("latest_phase", 1)
        print(build_state_text(metrics, phase))
        return 0

    if args.command == "propose":
        model_path = None if args.no_eval else Path(args.model)
        metrics = collect_metrics(
            project_root,
            model_path=model_path,
            n_episodes=args.episodes,
            obs_features=obs_features,
        )
        proposal = run_proposal(
            backend_name=args.backend,
            metrics=metrics,
            phase=args.phase,
            model_dir=args.model_dir,
            out_dir=PROPOSALS_DIR,
        )
        print(format_proposal(proposal))
        return 0

    if args.command == "apply":
        merged = apply_proposal(Path(args.proposal), OVERRIDES_PATH)
        print("Active overrides (coach/overrides.json):")
        print(format_overrides(merged))
        print(
            "\nNext: python scripts/train_ppo_curriculum.py --phase N "
            "--overrides coach/overrides.json"
        )
        return 0

    if args.command == "status":
        if OVERRIDES_PATH.exists():
            import json

            merged = json.loads(OVERRIDES_PATH.read_text())
            print(format_overrides(merged))
            print(f"\nsource: {merged.get('source', '<unknown>')}")
        else:
            print("No overrides active (coach/overrides.json does not exist).")
        return 0

    return 1


if __name__ == "__main__":
    exit(main() or 0)
