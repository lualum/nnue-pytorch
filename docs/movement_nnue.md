# Movement-based iterative evaluator

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
- pawns use fixed diagonal and forward directions, with the intervening state
  transforming the two-square forward message;
- bishops, rooks, and queens inject messages into ordered directional scans;
- all incoming messages are summed without counts or categorical annotations.

Ray propagation never checks occupancy and never terminates at an occupied
square. Each square first receives the current carrier and then applies the
same learned transition before the carrier proceeds:

```text
gate   = hard_sigmoid(W_path h_square + b_path)
altered = clamp(path_scale, -1, 1) * message
message_next = message + gate * (altered - message)
```

The initial embedding is the only place where empty and piece states are
identified. Consequently an empty square can learn to transmit most channels,
while any intervening piece state can learn a different transformation. There
are no explicit occupancy, termination, x-ray-strength, or piece-vacating
features in the ray code. The transition is linear in the carried message for a
fixed square state, so signals from multiple sources remain additive even when
they travel through the same line.

The update is a shared-weight gated MLP:

```text
candidate = hard_tanh(W_self h + W_message sum(messages) + W_state h_initial)
gate      = hard_sigmoid(G_self h + G_message sum(messages))
h_next    = h + gate * (candidate - h)
```

Three iterations and an 8-value state use fewer than 1,500 learned parameters.
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
3. rank, file, and diagonal squares whose sequential path transformation can
   depend on a dirty square;
4. the same expansion once per recurrent iteration.

States outside that cone are reused exactly. The Python implementation computes
batched message candidates before selecting dirty states, keeping training code
simple. The native Stockfish implementation applies the same dependency lists
in place and skips candidate calculations outside the cone.

The native runtime builds its sparse relationships from Stockfish's fixed
movement tables and ordered board directions. Each accumulator stack entry
stores both color perspectives: 64 piece codes and `iterations + 1` arrays of
64 small states.
Search computes only the requested side-to-move perspective and finds the most
recent cached state for that perspective when it is two plies back. The root
initializes both perspectives. The pooled result is multiplied by the exported
`nnue2score` before it is returned to search.

## Files and export

The implementation is in `model/modules/movement.py`, selected in
`model/model.py`, and trained through the normal `NNUE` wrapper. Save checkpoints
or `.pt` models normally. `serialize.py checkpoint.pt network.mnnue` writes a
portable little-endian runtime container with a versioned header and named,
shaped float32 tensors; deterministic movement geometry is not duplicated in
the file. The current native Stockfish runtime intentionally fixes the compact
default profile (dimension 8, three iterations); training can use 3–5 iterations
for experiments, but `.mnnue` export rejects a profile the engine cannot load.

Stockfish's existing `.nnue` binary schema describes a feature transformer plus
dense layer stacks and cannot represent a recurrent movement graph. The legacy
writer therefore rejects movement models instead of emitting a file that
Stockfish would misread. The native runtime reads the `.mnnue` tensor names,
which map one-to-one to the embeddings, message matrices, shared path/update
gates, and readout. Existing `.nnue` networks continue through the conventional
loader and evaluator unchanged.
