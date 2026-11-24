# CLI Commands

Available commands when using `uv run`:

| Command | Description |
|---------|-------------|
| `uv run pong` | Play against the expert predictive AI |
| `uv run pong --difficulty normal` | Play against an easier opponent |
| `uv run pong-play` | Play the game (same options as `pong`) |
| `uv run pong-train` | Train PPO agent (full curriculum) |
| `uv run pong-eval` | Evaluate trained agent |
| `uv run main.py` | Play the game (entry point) |
| `uv run -m pong` | Play the game (module) |

Difficulty choices are `slow`, `beginner`, `medium`, `normal`, `reactive`, and `expert`.

Use **UP/DOWN** or **W/S** to move, **Shift+S** for a faster reset, and **ESC** to quit.

To compare a saved PPO model with the scripted expert:

```bash
uv run python scripts/play_against_model.py --weights models/ppo_final.zip
uv run python scripts/play_against_model.py --opponent expert
```
