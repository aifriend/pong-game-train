"""
Proposal generation: turn one coaching session into a reviewable JSON proposal.

The decision model only picks candidates; this module translates those picks
into concrete, bounded config changes and records everything (state, questions,
answers with probabilities, derived diff) so a human can review before applying.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from pong.coach.engine import get_backend
from pong.coach.metrics import build_state_text, metrics_snapshot
from pong.coach.questions import (
    ADJUSTMENT_MULTIPLIERS,
    KNOB_BOUNDS,
    build_stage1_questions,
    build_stage2_questions,
)

INT_KNOBS = {"n_steps", "batch_size"}


def _adjust_value(key: str, old: float, adjustment: str) -> Optional[float]:
    """Apply a bounded adjustment to a knob value. None means no change."""
    multiplier = ADJUSTMENT_MULTIPLIERS.get(adjustment)
    if multiplier is None or old is None:
        return None

    # A percentage move from zero never yields a nonzero value
    if old == 0:
        return None

    new = old * multiplier
    lo, hi = KNOB_BOUNDS.get(key, (None, None))
    if lo is not None:
        new = max(lo, new)
    if hi is not None:
        new = min(hi, new)

    if key in INT_KNOBS:
        new = int(round(new))
        if new == int(old):
            return None
        return new

    # Two significant digits: coach proposals should be round numbers
    if new == 0:
        return 0.0
    from math import floor, log10

    rounded = round(new, -floor(log10(abs(new))) + 1)
    if rounded == old:
        return None
    return rounded


def derive_changes(
    answers: Dict[str, Any],
    phase: int,
    phase_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Translate the model's answers into concrete overrides for `phase`.

    Returns {"overrides": {...}, "changes": [...], "warnings": [...]}.
    """
    overrides: Dict[str, Any] = {}
    changes = []
    warnings = []

    for knob_q, adj_q, kind in (
        ("reward_knob", "reward_adjustment", "reward"),
        ("hp_knob", "hp_adjustment", "hyperparameter"),
    ):
        knob = answers.get(knob_q, {}).get("choice", "none")
        adjustment = answers.get(adj_q, {}).get("choice", "keep")
        if knob == "none" or adjustment == "keep":
            continue

        old = phase_config.get(knob)
        new = _adjust_value(knob, old, adjustment)
        if new is None:
            warnings.append(
                f"{kind}: {knob} {adjustment} had no effect "
                f"(already at bound or unchanged)"
            )
            continue

        overrides[knob] = new
        changes.append(
            {
                "kind": kind,
                "phase": phase,
                "key": knob,
                "old": old,
                "new": new,
                "confidence": answers.get(knob_q, {})
                .get("probabilities", {})
                .get(knob),
            }
        )

    feature = answers.get("obs_feature", {}).get("choice", "none")
    if feature != "none":
        existing = list(phase_config.get("obs_features") or [])
        if feature not in existing:
            existing.append(feature)
            overrides["obs_features"] = existing
            changes.append(
                {
                    "kind": "observation",
                    "phase": phase,
                    "key": "obs_features",
                    "old": phase_config.get("obs_features"),
                    "new": existing,
                    "confidence": answers.get("obs_feature", {})
                    .get("probabilities", {})
                    .get(feature),
                }
            )
            warnings.append(
                "obs_features changes the observation shape: phase training "
                "restarts from a fresh model, earlier checkpoints cannot be resumed."
            )
        else:
            warnings.append(f"obs feature {feature} already enabled; no change")

    return {"overrides": overrides, "changes": changes, "warnings": warnings}


def run_proposal(
    backend_name: str,
    metrics: Dict[str, Any],
    phase: Optional[int] = None,
    model_dir: Optional[str] = None,
    out_dir: Path = Path("coach/proposals"),
) -> Dict[str, Any]:
    """
    Run one coaching session and write the proposal JSON.

    Returns the proposal dict. The target phase defaults to the latest phase
    found in the collected metrics.
    """
    if phase is None:
        phase = metrics.get("latest_phase")
        if phase is None:
            raise ValueError(
                "No phase given and none found in metrics; pass --phase explicitly"
            )

    phase_config = metrics.get("config", {}).get(phase, {})
    state = build_state_text(metrics, phase)

    backend = get_backend(backend_name, model_dir)

    # Stage 1: condition checks, knob selection, obs feature, phase health.
    # Stage 2: adjustment magnitudes, asked only for the knobs chosen in
    # stage 1, with the chosen knob named in the question so the magnitude
    # decision is conditioned on it. Both stages share one backend session so
    # the model loads from disk only once.
    try:
        stage1_questions = build_stage1_questions(phase)
        answers = backend.predict(state, stage1_questions)

        stage2_questions = build_stage2_questions(answers)
        if stage2_questions:
            stage2_state = state + (
                "\n\nDecisions already made: "
                f"reward knob = {answers.get('reward_knob', {}).get('choice')}, "
                f"hyperparameter = {answers.get('hp_knob', {}).get('choice')}, "
                f"observation feature = {answers.get('obs_feature', {}).get('choice')}."
            )
            answers.update(backend.predict(stage2_state, stage2_questions))
    finally:
        backend.close()

    questions = {**stage1_questions, **stage2_questions}

    derived = derive_changes(answers, phase, phase_config)

    proposal = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend.name,
        "phase": phase,
        "state": state,
        "metrics_snapshot": metrics_snapshot(metrics, phase),
        "questions": questions,
        "answers": answers,
        "derived_changes": derived["changes"],
        "warnings": derived["warnings"],
        "overrides": {"phases": {str(phase): derived["overrides"]}},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"proposal_{stamp}_phase{phase}_{backend.name}.json"
    path.write_text(json.dumps(proposal, indent=2, default=str))
    proposal["path"] = str(path)

    return proposal


def format_proposal(proposal: Dict[str, Any]) -> str:
    """Human-readable rendering of a proposal for review."""
    lines = [
        f"Coach proposal ({proposal['backend']}) for phase {proposal['phase']}",
        f"created {proposal['created_at']}",
        "",
        "Condition checks (P(true)):",
    ]
    for gate in ("plateaued", "rally_farming", "value_diverging", "entropy_collapsed"):
        answer = proposal["answers"].get(gate)
        if answer is not None:
            lines.append(f"  {gate}: {answer['noul']*100:.0f}%")

    health = proposal["answers"].get("phase_health")
    if health:
        lines.append(
            f"\nPhase health score: {health['score']:.2f} / 3 "
            f"(0=failing, 3=on_track)"
        )

    lines.append("\nProposed changes:")
    if proposal["derived_changes"]:
        for change in proposal["derived_changes"]:
            conf = change.get("confidence")
            conf_str = f" (confidence {conf*100:.0f}%)" if conf is not None else ""
            lines.append(
                f"  [{change['kind']}] {change['key']}: "
                f"{change['old']!r} -> {change['new']!r}{conf_str}"
            )
    else:
        lines.append("  (none - the coach recommends keeping the current config)")

    for warning in proposal.get("warnings", []):
        lines.append(f"  ⚠️  {warning}")

    lines.append(f"\nSaved to {proposal.get('path', '<not saved>')}")
    return "\n".join(lines)
