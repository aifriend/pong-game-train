"""
Decision backends for the RL coach.

A backend takes a state string and the questions dict from questions.py and
returns the answers dict in the NeoHorse-Jev shape:

    {question_id: {"choice": key, "probabilities": {...}}        # Choice
                  {"noul": p_true}                               # Noul
                  {"score": expected_rating, "probabilities": {...}}}  # Score

NeoHorseBackend drives the real neohorse_decision runtime through a persistent
subprocess (scripts/coach_worker.py) so the model loads once per coaching
session even though a session issues two decision stages. Call close() when
done (run_proposal does this in a finally block).

MockBackend is deterministic heuristics for tests and dry runs.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_WORKER_VENV = ".venv-neohorse"


class NeoHorseBackend:
    """Backend backed by NeoHorse-Jev-4B via the neohorse_decision runtime."""

    name = "neohorse"

    def __init__(
        self,
        model_dir: str,
        device: Optional[str] = None,
        worker_python: Optional[str] = None,
    ):
        if worker_python is None:
            default = (
                Path(__file__).resolve().parents[2]
                / DEFAULT_WORKER_VENV
                / "bin"
                / "python"
            )
            worker_python = str(default)
        if not Path(worker_python).exists():
            raise ValueError(
                f"NeoHorse worker interpreter not found at {worker_python}; "
                "create .venv-neohorse per docs/COACH.md or pass worker_python"
            )
        self.worker_python = worker_python
        self.model_dir = model_dir
        self.device = device  # None = let the worker auto-detect (mps > cuda > cpu)
        self._proc: Optional[subprocess.Popen] = None

    def _ensure_worker(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return

        worker_script = str(
            Path(__file__).resolve().parents[2] / "scripts" / "coach_worker.py"
        )
        env = dict(os.environ)
        env["NEOHORSE_MODEL_DIR"] = str(self.model_dir)
        if self.device:
            env["NEOHORSE_DEVICE"] = self.device

        self._proc = subprocess.Popen(
            [self.worker_python, worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # model load progress goes to the parent's stderr
            text=True,
            env=env,
        )

        # The first line arrives only after the model is loaded
        ready = self._proc.stdout.readline()
        if not ready:
            raise RuntimeError("NeoHorse worker exited before reporting ready")
        try:
            status = json.loads(ready)
        except json.JSONDecodeError:
            raise RuntimeError(f"NeoHorse worker sent a malformed banner: {ready!r}")
        if "error" in status:
            raise RuntimeError(f"NeoHorse worker failed to load: {status['error']}")
        self.device = status.get("device", self.device)

    def predict(self, state: str, questions: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_worker()
        assert self._proc is not None

        try:
            self._proc.stdin.write(
                json.dumps({"state": state, "questions": questions}) + "\n"
            )
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            self.close()
            raise RuntimeError(
                f"NeoHorse worker died before the request could be sent: {e}"
            )

        line = self._proc.stdout.readline()
        if not line:
            self.close()
            raise RuntimeError("NeoHorse worker exited mid-request")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(f"NeoHorse worker error:\n{response['error']}")
        return response["answers"]

    def close(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
            finally:
                self._proc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class MockBackend:
    """
    Deterministic stand-in for development and tests.

    Answers every Noul with 0.5, picks the first non-"none" candidate for
    reward/hp knobs, "keep" for adjustments, "none" for obs features, and the
    middle health level. Probabilities are uniform.
    """

    name = "mock"

    def predict(self, state: str, questions: Dict[str, Any]) -> Dict[str, Any]:
        answers: Dict[str, Any] = {}
        for qid, question in questions.items():
            qtype = question["type"]
            if qtype == "noul":
                answers[qid] = {"noul": 0.5}
            elif qtype == "choice":
                candidates = list(question["criteria"])
                if qid in ("reward_knob", "hp_knob"):
                    pick = next(c for c in candidates if c != "none")
                elif qid.endswith("_adjustment"):
                    pick = "keep"
                else:
                    pick = "none"
                n = len(candidates)
                answers[qid] = {
                    "choice": pick,
                    "probabilities": {c: round(1.0 / n, 4) for c in candidates},
                }
            elif qtype == "score":
                levels = question["criteria"]
                n = len(levels)
                mid = (n - 1) / 2.0
                answers[qid] = {
                    "score": mid,
                    "probabilities": {str(i): round(1.0 / n, 4) for i in range(n)},
                }
            else:
                raise ValueError(f"Unknown question type: {qtype}")
        return answers

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def get_backend(name: str, model_dir: str = None, device: str = None):
    if name == "mock":
        return MockBackend()
    if name == "neohorse":
        if not model_dir:
            raise ValueError("--model-dir is required for the neohorse backend")
        return NeoHorseBackend(model_dir, device=device)
    raise ValueError(f"Unknown backend: {name}")
