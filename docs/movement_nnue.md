# Movement-based iterative evaluator

For the direct-ray ablation and paired training protocol, see
[Direct-ray experiment](ordered_rays.md).

The `movement` network is a small alternative to the conventional feature
transformer and dense layer stacks. It keeps the existing sparse training batch
and scalar loss interface, but decodes the `HalfKAv2_hm^` component into 64
side-to-move-normalized square states before evaluation.

Enable it with:

```sh
python train.py data.binpack \
  --features HalfKAv2_hm^ \
  --network-type movement \
  --movement-dim 8 \
  --movement-iterations 3
```

The model requires a `HalfKAv2_hm^` component. If the configured feature set is
composed, the movement model automatically asks the native loader for HalfKA
alone: threat and pawn-pair tables are redundant with its movement messages.

## Network

Each square begins with a learned piece-state embedding (empty plus 12 relative
piece codes) and a learned square embedding. The position is normalized to the
side to move: our pawns always travel toward increasing ranks, and the output is
already in Stockfish's negamax perspective.

Every iteration adds messages along deterministic chess geometry:

- knights and kings use fixed sparse jump edge lists;
- pawns use fixed diagonal and forward directions, including the state of the
  one intervening square for a double push;
- bishops, rooks, and queens send a direct message to every square on their
  geometric ray;
- all incoming messages are summed without counts or categorical annotations.

Each direct slider relation contains its source and target square states,
direction, distance, clipped blocker count, and the states of the first two
occupied squares strictly between source and target. These blocker slots are
selected from a fixed `[relation, path-square]` tensor in parallel. For a rook
on `a1`, pawn on `a3`, and queen on `a7`, the `a1 -> a7` relation includes the
pawn as its first blocker in the same layer.

```text
message = W_source h_source + W_target h_target
        + W_first h_first_blocker + W_second h_second_blocker
        + E_direction + E_distance + E_blocker_count
```

The network is told only primitive ray geometry and which intervening squares
are occupied. It is not given pin, x-ray, or battery labels. The source-to-
target relation means x-rays do not require an earlier message to be preserved
through each blocker.

The update is a shared-weight gated MLP:

```text
candidate = hard_tanh(W_self h + W_message sum(messages) + W_state h_initial)
gate      = hard_sigmoid(G_self h + G_message sum(messages))
h_next    = h + gate * (candidate - h)
```

Three iterations and an 8-value state remain compact.
There are no Q/K/V projections and no dense 64 by 64 attention or adjacency
tensor. A mean over the 64 final square states feeds a tiny scalar readout in the
same units expected by the existing NNUE loss; `nnue2score` converts it to the
centipawn-like search score.

## Incremental search state

`MovementEvaluationNetwork.create_accumulator()` caches the state after every
iteration for one fixed perspective. The `NNUEModel` search-facing helpers keep
both a White and Black perspective cache, just as Stockfish's conventional NNUE
does, so the side-to-move flip does not invalidate every square after each ply.
`update_accumulator()` compares the next 64 piece codes for each fixed
perspective with the cached position, then updates a conservative dependency
cone:

1. changed source/destination squares;
2. jump and pawn destinations of dirty squares;
3. rank, file, and diagonal squares whose direct ray relation can include a
   dirty source, target, or blocker;
4. the same expansion once per recurrent iteration.

States outside that cone are reused exactly. The Python implementation computes
batched message candidates before selecting dirty states, keeping training code
simple. The native Stockfish implementation applies the same dependency lists
in place and skips candidate calculations outside the cone.

The current native runtime implements the older sequential-ray model and is
therefore incompatible with direct-ray checkpoints. A native implementation
must construct the same fixed relation list, use parallel blocker extraction,
and cross-check every score before engine games.

## Files and export

The implementation is in `model/modules/movement.py`, selected in
`model/model.py`, and trained through the normal `NNUE` wrapper. Save checkpoints
or `.pt` models normally. Direct-ray models cannot currently be written as
`.mnnue`, because version 2 describes the old sequential-ray runtime.

Stockfish's existing `.nnue` binary schema describes a feature transformer plus
dense layer stacks and cannot represent this graph. Both export paths reject
direct-ray models rather than producing a file Stockfish would misread.
