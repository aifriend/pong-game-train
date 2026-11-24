#!/usr/bin/env python3
"""
Persistent subprocess worker that runs the NeoHorse-Jev-4B native runtime.

The neohorse_decision package lives in .venv-neohorse (heavy torch/transformers
stack kept out of the project venv), so the coach project code cannot import it
directly. pong.coach.engine.NeoHorseBackend launches this script with the
worker venv's interpreter and keeps it alive for the whole coaching session,
so the model is loaded from disk only once.

Configuration comes from environment variables:
    NEOHORSE_MODEL_DIR  path to the model bundle (required)
    NEOHORSE_DEVICE     torch device (optional; auto-detects mps > cuda > cpu)

Protocol: after loading the model the worker prints a {"ready": true} line,
then reads one request per stdin line ({"state": str, "questions": dict}) and
writes one {"answers": {...}} or {"error": traceback} line per request.
"""

import json
import os
import sys
import traceback


def main() -> int:
    from neohorse_decision import DecisionEngine

    device = os.environ.get("NEOHORSE_DEVICE")
    if not device:
        import torch

        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )

    engine = DecisionEngine(os.environ["NEOHORSE_MODEL_DIR"], device=device)
    print(json.dumps({"ready": True, "device": device}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            result = engine.predict(
                {
                    "state": request["state"],
                    "questions": request["questions"],
                }
            )
            out = {"answers": result["answers"]}
        except Exception:
            out = {"error": traceback.format_exc(limit=3)}
        print(json.dumps(out), flush=True)

    return 0


if __name__ == "__main__":
    exit(main() or 0)
