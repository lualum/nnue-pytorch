# RayGNN v3 reference

`RayGNN()` implements [RayGNN v3](../RayGNN_Design_v3.md), Part I. It is a standalone, jointly trained White-positive evaluator in pawn units. It uses 64 square nodes, two 96-wide graph layers, three exclusive relationship classes, 32-wide incoming and outgoing messages, and a learned piece-square, contextual, and spatial output. Engine search, terminal adjudication, history-dependent draws, and mate-range clamping remain outside the model.

```python
import chess
from raygnn import RayGNN, boards_to_batch

batch = boards_to_batch([chess.Board()])
value_white = RayGNN()(batch)  # [1,1], pawn units
```

The direct signature is `model(piece, side_to_move, castling, en_passant, draw_state=None)`. `piece` is `[B,64]` with empty=0, White P/N/B/R/Q/K=1..6, Black=7..12; side to move is `[B,1]` with +1 White and -1 Black; castling is `[B,4]` in WK/WQ/BK/BQ order; en-passant is `[B]` with the FEN target 0..63 or 64 for none. Squares run from a1=0 to h8=63. The board adapter preserves the FEN en-passant target even when no legal capture exists.

`RayGNNEvaluator.from_checkpoint("best.pt").evaluate_cp([board])` returns rounded White-positive centipawns. Pass `side_to_move=True` for engine-relative scores and `max_cp` to clamp below an engine's reserved mate range. V2 checkpoints are incompatible with v3.

Training records are JSONL with `fen`, White-positive `eval_cp`, and `game_id` or `opening_family`. Mate records are excluded. `python -m raygnn.train positions.jsonl --output best.pt` runs a five-million-position seed-17 screen with a 20-million-position schedule, `eval_cp / 100` targets, Huber delta 1, AdamW, and gradient clipping at 1. Use `--epochs 20 --resume best.latest.pt` to continue the same seed. The training script is a simple reference runner; the full correctness gates, fixed validation protocol, controls, and allocation plan are in the v3 design.

`--variant` selects `no_message`, `no_relation_separation`, `dense_context`, `king_relative`, `wider_spatial`, `wider_messages`, `global_pooling`, `three_layers`, or `simple_pair`. `--augment-color` enables rank reflection and color swap. `--edge-chunk-size` limits pair computation chunk size. `--draw-state` appends halfmove clock divided by 100 and twofold repetition; records must provide `repetition_twofold`, because FEN does not encode history. Keep the same input contract at inference.

`python -m raygnn.benchmark --device cpu --iterations 20` times graph feature construction and full forward inference. It does not launch a training sweep.
