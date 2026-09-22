# Direct-ray experiment

## Motivation

Sequential ray propagation makes a long-range relation depend on every
intermediate state preserving a useful signal. This is a poor fit for x-rays:
in `rook -> pawn -> queen`, the rook-to-queen interaction must survive several
separate updates before it exists. It also makes the result sensitive to an
implementation order along the ray.

The movement evaluator now builds every directed slider relation at once. A
relation has a source square, target square, direction, distance, the number of
occupied squares strictly between them, and the first two occupied blocker
states in board order. The relation for a rook on `a1` and queen on `a7` exists
whether `a3` is empty or occupied. If a pawn occupies `a3`, it becomes the
first blocker context of the direct `a1 -> a7` message.

This is a sparse tensor of about 1,500 directed source-target pairs, not dense
attention. Blocker extraction is performed as gather and reduction operations
over padded fixed path tensors, with no recurrent carrier or directional scan.
The first two slots preserve the most immediately relevant x-ray structure;
the clipped count reports whether further blockers exist. A later ablation can
test three or more slots.

`--movement-ordered-rays` remains accepted only for old scripts. It no longer
changes the architecture because direct rays are always used.

## Training and comparison

Use the normal trainer without the compatibility flag:

```sh
python train.py train.binpack --validation-datasets valid.binpack \
  --validation-size 100000 --features 'HalfKAv2_hm^' \
  --network-type movement --movement-dim 8 --movement-iterations 3 \
  --batch-size 256 --epoch-size 1000000 --max-epochs 20 --seed 42 \
  --default-root-dir runs/direct-rays

python -m pytest tests/test_movement_network.py tests/test_direct_rays.py -q
```

Use game-disjoint training and validation files. Compare against the last
sequential branch at equal position budgets and equal wall time, then inspect a
tactical x-ray subset separately. A lower general loss is necessary but not
sufficient evidence that the extra relation context helps search.

Direct-ray checkpoints must remain `.ckpt` or `.pt` files. The existing native
runtime and `.mnnue` format encode the old sequential-ray graph. Native
execution requires a matching relation-list implementation followed by
PyTorch-to-engine parity checks.

## Tests

Tests verify that distant x-ray targets receive first and second blocker state
in one call, a blocker changes the direct relation without recurrent path state,
incremental evaluation matches full recomputation after edits, and checkpoint
round-trip training works. The exporter rejects this incompatible layout.
