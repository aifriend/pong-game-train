"""
Metrics collection for the RL coach.

Gathers what the decision model needs to reason about training:
- per-phase TensorBoard scalars (win rate, losses, entropy, KL),
- an optional fresh evaluation of a saved model against the scripted opponents,
- the current curriculum configuration.

Everything is reduced to a compact text summary (`build_state_text`) that fits
comfortably in the model's context.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

# TensorBoard tags the coach reasons about. Missing tags are skipped.
TB_TAGS = [
    "rollout/win_rate",
    "curriculum/win_rate",
    "train/entropy_loss",
    "train/value_loss",
    "train/policy_gradient_loss",
    "train/approx_kl",
    "train/clip_fraction",
    "train/explained_variance",
]

# How many of the most recent scalar points to summarize per tag.
TAIL = 40


def _summarize_tag(values: List[float]) -> Dict[str, Any]:
    """Summarize a scalar series into last value, trend, and a compact sample."""
    if not values:
        return {}
    tail = values[-TAIL:]
    half = max(len(tail) // 2, 1)
    first_half = sum(tail[:half]) / half
    second_half = sum(tail[-half:]) / half
    return {
        "last": tail[-1],
        "mean": sum(tail) / len(tail),
        "trend": second_half - first_half,
        "n_points": len(values),
        "sample": [round(v, 4) for v in tail[:: max(len(tail) // 8, 1)]],
    }


def collect_tb_metrics(tensorboard_root: Path) -> Dict[str, Dict[str, Any]]:
    """
    Read per-phase scalar summaries from one tensorboard run directory.

    Expects subdirectories named like phase1_0, phase2_0, ... each containing
    event files. Returns {phase_number: {tag: summary}}.
    """
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    phases: Dict[str, Dict[str, Any]] = {}
    phase_dirs = sorted(
        (d for d in tensorboard_root.iterdir() if d.is_dir() and d.name.startswith("phase")),
        key=lambda d: d.name,
    )
    for phase_dir in phase_dirs:
        phase_num = phase_dir.name.split("_")[0].replace("phase", "")
        accumulator = EventAccumulator(str(phase_dir))
        accumulator.Reload()
        tags = {}
        for tag in TB_TAGS:
            if tag not in accumulator.Tags().get("scalars", []):
                continue
            events = accumulator.Scalars(tag)
            summary = _summarize_tag([e.value for e in events])
            if summary:
                tags[tag] = summary
        if tags:
            phases[phase_num] = tags
    return phases


def find_latest_tensorboard_dir(project_root: Path) -> Optional[Path]:
    """Return the most recently modified tensorboard_* directory, if any."""
    candidates = [
        d
        for d in project_root.iterdir()
        if d.is_dir() and (d.name == "tensorboard" or d.name.startswith("tensorboard_"))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


def collect_evaluation(
    model_path: Path,
    n_episodes: int = 20,
    obs_features: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run a fresh evaluation of a saved model against the scripted opponents."""
    from stable_baselines3 import PPO

    from pong.env.pong_headless import OpponentType
    from scripts.train_ppo_curriculum import evaluate_model

    model = PPO.load(str(model_path))
    expected_dim = 9 + len(obs_features or [])
    model_dim = model.observation_space.shape[0]
    if model_dim != expected_dim:
        raise ValueError(
            f"Model expects a {model_dim}-dim observation but evaluation was "
            f"configured for {expected_dim} dims (obs_features={obs_features}). "
            "Pass --obs-features with the same features the model was trained "
            "with (see the proposal that enabled them)."
        )

    results = {}
    for opponent in [
        OpponentType.SLOW_AI,
        OpponentType.BEGINNER_AI,
        OpponentType.NORMAL_AI,
        OpponentType.REACTIVE_AI,
    ]:
        result = evaluate_model(
            model,
            opponent,
            n_episodes=n_episodes,
            deterministic=False,
            verbose=False,
            obs_features=obs_features,
        )
        results[opponent.value] = {
            "win_rate": result["win_rate"],
            "avg_score_diff": result["avg_score_diff"],
            "unfinished_rate": result["unfinished_rate"],
        }
    return results


def current_config_snapshot() -> Dict[int, Dict[str, Any]]:
    """Coach-relevant subset of the live CURRICULUM_PHASES config."""
    from scripts.train_ppo_curriculum import CURRICULUM_PHASES

    keys = (
        "hit_reward",
        "tracking_reward",
        "pressure_scale",
        "step_penalty",
        "learning_rate",
        "ent_coef",
        "timesteps",
        "ball_speed",
        "target_win_rate",
        "obs_features",
    )
    # PPO constructor knobs are not part of CURRICULUM_PHASES; the trainer
    # falls back to these defaults. Snapshot the effective values so the coach
    # can reason about (and adjust) them too.
    ppo_defaults = {"n_steps": 1024, "batch_size": 128, "clip_range": 0.2}

    snapshot = {}
    for phase, config in CURRICULUM_PHASES.items():
        entry = {k: config[k] for k in keys if k in config}
        for key, default in ppo_defaults.items():
            entry[key] = config.get(key, default)
        snapshot[phase] = entry
    return snapshot


def collect_metrics(
    project_root: Path,
    model_path: Optional[Path] = None,
    n_episodes: int = 20,
    obs_features: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Collect the full metrics picture for one coaching session.

    Reads the latest tensorboard_* directory and, when model_path points to an
    existing model, runs a fresh evaluation.
    """
    metrics: Dict[str, Any] = {"config": current_config_snapshot()}

    tb_dir = find_latest_tensorboard_dir(project_root)
    if tb_dir is not None:
        metrics["tensorboard_dir"] = tb_dir.name
        metrics["phases"] = collect_tb_metrics(tb_dir)

    if model_path is not None:
        path = Path(model_path)
        if path.suffix != ".zip":
            path = Path(str(path) + ".zip")
        if path.exists():
            metrics["model"] = str(path)
            metrics["evaluation"] = collect_evaluation(path, n_episodes, obs_features)

    if "phases" in metrics and metrics["phases"]:
        metrics["latest_phase"] = max(int(p) for p in metrics["phases"])

    return metrics


def build_state_text(metrics: Dict[str, Any], phase: int) -> str:
    """
    Render the metrics as the compact natural-language `state` the decision
    model reasons over. Numbers are rounded; the full JSON goes into the
    proposal file for the human review.
    """
    lines = [
        "Pong PPO agent training status. The agent plays right-side Pong against "
        "scripted AI opponents of increasing speed. Goal: maximize game win rate.",
        f"Coaching focuses on phase {phase}.",
        "",
        "Current configuration for this phase:",
    ]

    config = metrics.get("config", {}).get(phase, {})
    for key, value in config.items():
        lines.append(f"  {key} = {value}")

    phases = metrics.get("phases", {})
    phase_tb = phases.get(str(phase))
    if phase_tb:
        lines += ["", "Recent training metrics (mean over last ~40 logged points):"]
        for tag, summary in phase_tb.items():
            lines.append(
                f"  {tag}: last={summary['last']:.4f} mean={summary['mean']:.4f} "
                f"trend={summary['trend']:+.4f}"
            )
        win_rate = phase_tb.get("rollout/win_rate") or phase_tb.get("curriculum/win_rate")
        if win_rate:
            lines.append(f"  win rate sample: {win_rate['sample']}")

    evaluation = metrics.get("evaluation")
    if evaluation:
        lines += ["", "Fresh evaluation against scripted opponents (sampled actions):"]
        for opponent, result in evaluation.items():
            lines.append(
                f"  vs {opponent}: {result['win_rate']*100:.0f}% win, "
                f"avg score diff {result['avg_score_diff']:+.2f}, "
                f"{result['unfinished_rate']*100:.0f}% unfinished"
            )

    return "\n".join(lines)


def metrics_snapshot(metrics: Dict[str, Any], phase: int) -> Dict[str, Any]:
    """
    Structured metrics summary embedded in a proposal, so a reviewer sees the
    same evidence the model saw without re-running collect.
    """
    snapshot = {
        "tensorboard_dir": metrics.get("tensorboard_dir"),
        "latest_phase": metrics.get("latest_phase"),
        "phase_tb": metrics.get("phases", {}).get(str(phase)),
        "model": metrics.get("model"),
        "evaluation": metrics.get("evaluation"),
    }
    return {k: v for k, v in snapshot.items() if v is not None}
