"""
Apply an approved coach proposal to coach/overrides.json.

The overrides file is consumed by train_ppo_curriculum.py --overrides and is
gitignored. Source files are never edited; deleting the overrides file returns
training to the default curriculum.

Later proposals overwrite the same keys; keys they do not mention are kept.
"""

import json
from pathlib import Path
from typing import Any, Dict

from scripts.train_ppo_curriculum import OVERRIDABLE_KEYS

DEFAULT_OVERRIDES_PATH = Path("coach/overrides.json")


def load_proposal(path: Path) -> Dict[str, Any]:
    proposal = json.loads(Path(path).read_text())
    if "overrides" not in proposal:
        raise ValueError(f"Proposal {path} has no 'overrides' section")
    return proposal


def validate_overrides(overrides: Dict[str, Any]) -> None:
    """Reject proposals whose overrides touch keys the trainer does not honor."""
    for phase, values in overrides.get("phases", {}).items():
        unknown = set(values) - OVERRIDABLE_KEYS
        if unknown:
            raise ValueError(
                f"Override for phase {phase} has keys {sorted(unknown)} "
                f"that the trainer cannot honor"
            )


def apply_proposal(
    proposal_path: Path,
    overrides_path: Path = DEFAULT_OVERRIDES_PATH,
) -> Dict[str, Any]:
    """
    Merge a proposal's overrides into the active overrides file and return the
    merged document with provenance metadata.
    """
    proposal = load_proposal(proposal_path)
    overrides = proposal["overrides"]
    validate_overrides(overrides)

    path = Path(overrides_path)
    if path.exists():
        existing = json.loads(path.read_text())
    else:
        existing = {"phases": {}}

    for phase, values in overrides.get("phases", {}).items():
        existing["phases"].setdefault(phase, {}).update(values)

    existing["source"] = str(proposal_path)
    existing["updated_at"] = proposal.get("created_at")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2))
    return existing


def format_overrides(overrides: Dict[str, Any]) -> str:
    """Human-readable rendering of the active overrides."""
    lines = []
    for phase, values in sorted(overrides.get("phases", {}).items(), key=lambda kv: int(kv[0])):
        for key, value in values.items():
            lines.append(f"  phase {phase}: {key} = {value}")
    return "\n".join(lines) if lines else "  (no overrides active)"
