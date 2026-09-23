# RayGNN reference implementation

`RayGNN()` implements the v0.1 widths in [the design document](../RayGNN_Design_Document.md): three 64-wide layers, eight 16-wide ray channels, 16-wide knight and pawn channels, an 832-wide structured readout, a 32-wide state branch, a 64-wide raw-board skip, and a 928 → 256 → 64 → 1 value head. Values are White-positive pawn units, with deterministic material plus a learned correction. The optional WDL logits are ordered White/Draw/Black.

Input square order is python-chess A1=0 through H8=63. Piece IDs are empty=0, White pawn through king=1..6, Black pawn through king=7..12. `boards_to_batch` includes side to move, K/Q castling rights for each color, legal en-passant target, halfmove clock, and a twofold repetition flag. The engine remains responsible for terminal results and draw adjudication. Input validation expects exactly one king of each color. The default model averages a position with its color-swapped 180-degree rotation to guarantee value negation under that transform. This doubles neural inference work; set `enforce_color_symmetry=False` to benchmark the cheaper unsymmetrized variant.

```python
import chess
from raygnn import RayGNN, boards_to_batch

batch = boards_to_batch([chess.Board()])
result = RayGNN()(batch)
print(result.value, result.material, result.correction)
```

`RayGNNEvaluator.from_checkpoint("best.pt").evaluate_cp([board])` returns White-positive centipawns for an engine adapter. The search engine must handle checkmate, stalemate, and draw rules before requesting a neural evaluation.

Train with `python -m raygnn.train positions.jsonl --output best.pt`. Each JSONL record must have `fen`, White-positive `eval_cp`, and `game_id` or `opening_family` for grouped holdout. Mate records with a `mate` field are skipped. Extreme scores are smoothly mapped to `10*tanh(eval_cp/1000)` pawn units. Optional `wdl` is a three-element White/Draw/Black weight vector; pass `--wdl-head` only when all records have it. Ablation flags are `--layers 0..4`, `--direct-only`, `--first-piece-only`, `--simple-readout`, and `--no-raw-skip`. For matched experiments, use identical data and settings apart from the selected ablation.

Run `python -m raygnn.benchmark --device cpu --iterations 20` for geometry time and full forward latency. Single-position and batched benchmarks include occupancy-dependent feature construction in the full timing. The Python reference favors clear semantics over production search latency; integration into a search engine requires a separate API and profiling pass.
