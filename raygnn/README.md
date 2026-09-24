# RayGNN second-design reference

`RayGNN()` implements [RayGNN Design Document 2nd](../RayGNN_Design_Document_2nd.md), Part I: 64 squares, 1,792 directed edges, two independent graph layers with 32-wide messages, a 128-wide mean-pooled square readout, the exact 832-wide board one-hot bypass, and a `976 → 64 → 1` value head. Output is an unbounded White-positive scalar in pawn units. Legal moves, terminal scores, draw adjudication, and mate-range clamping belong to the engine.

```python
import chess
from raygnn import RayGNN, boards_to_batch

batch = boards_to_batch([chess.Board()])
value_white = RayGNN()(batch)  # [1,1], pawn units
```

The direct forward signature is `model(piece, side_to_move, castling, en_passant, draw_state=None)`. `piece` is `[B,64]` with empty=0, White P/N/B/R/Q/K=1..6, Black=7..12; side to move is `[B,1]` with +1 White and -1 Black; castling is `[B,4]` in WK/WQ/BK/BQ order; en-passant is `[B]` with FEN target 0..63 or 64 for none. The board adapter preserves a FEN en-passant target even when no legal capture exists.

`RayGNNEvaluator.from_checkpoint("best.pt").evaluate_cp([board])` returns rounded White-positive centipawns. Pass `side_to_move=True` for engine-relative scores and `max_cp` to clamp below the engine's reserved mate range.

Training records are JSONL with `fen`, White-positive `eval_cp`, and `game_id` (or `opening_family`). Mate records are excluded. `python -m raygnn.train positions.jsonl --output best.pt` screens the reference for 20 fixed-size epochs of one million sampled positions, seed 17, using a schedule set to 50 epochs. Use `--epochs 50 --resume best.latest.pt` to continue the same seed, then run seeds 29 and 43 separately. The default `existing_tanh` target preserves the original trainer's target scaling for matched comparisons; select `--target-transform cp_pawns` for a fresh `eval_cp / 100` pipeline. Both use Huber delta 1, AdamW and gradient clipping at 1. The best and latest checkpoints contain architecture, input schema, perspective, target transform and feature ordering.

`--variant` selects the optional Part II architecture controls and one-change variants: `original_compact`, `no_message`, `raw_board_only`, `channelwise_gates`, `king_relative`, `smaller_messages`, `smaller_readout`, `relation_biases`, `one_layer`, or `three_layers`. `--augment-color` enables rank reflection and color swap. `--float32-reductions`, `--edge-chunk-size`, and `--sparse-board-projection` are implementation options. `--draw-state` appends halfmove clock divided by 100 and twofold repetition flag; training records must provide `repetition_twofold` because FEN does not encode history. Keep the same input contract at inference.

`python -m raygnn.benchmark --device cpu --iterations 20` times feature construction and full forward inference. Experiment budgets and selection order are in the design document; the command does not launch a multi-run sweep automatically.
