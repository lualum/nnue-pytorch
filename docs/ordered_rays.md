# Ordered-ray mixing experiment

## Motivation

The spatial evaluator proposal calls for blocker identity and order to affect
long-range interactions. The existing movement evaluator already transports
signals through learned square states, but each transition is diagonal:
`c_next = f(h) * c`. For fixed intervening states A and B, their transformations
commute. A single source's signal cannot distinguish A-then-B from B-then-A.
This is a limitation of the path operator, not a claim that the whole network
is invariant to swapping pieces: square embeddings, injected signals and
subsequent iterations can already distinguish many such positions.

Enable the experiment using `--movement-ordered-rays`. The transition becomes

```
W = path_mix / max(1, row_sum(abs(path_mix)))
c_next = f(h) * c + (1 - abs(f(h))) * (W @ c)
```

These state-dependent matrices need not commute. The infinity norm of a
transported signal cannot grow at a transition, since each matrix row has
absolute sum at most one. New source injections can still increase the total
carrier. The operation remains linear in the carrier, so multiple sliders can
share the existing directional scan. The pawn double-step uses the same
transition for consistency. Empty squares can also transform signals; no
hand-coded pin, x-ray, or blocker labels are introduced.

The mixer starts at zero, exactly reproducing the baseline for identical
remaining weights while receiving gradients immediately. Default dimension 8
adds 64 parameters. Work increases by a small matrix multiply per ray step;
parameter count alone does not imply faster evaluation. The original option
remains the default and retains its checkpoint tensor layout.

## Training and comparison

Use the existing binpack loader, optimizer, loss and checkpoint path:

```sh
python train.py train.binpack --validation-datasets valid.binpack \
  --validation-size 100000 --features 'HalfKAv2_hm^' \
  --network-type movement --movement-dim 8 --movement-iterations 3 \
  --batch-size 256 --epoch-size 1000000 --max-epochs 20 --seed 42 \
  --default-root-dir runs/ray-baseline

python train.py train.binpack --validation-datasets valid.binpack \
  --validation-size 100000 --features 'HalfKAv2_hm^' \
  --network-type movement --movement-dim 8 --movement-iterations 3 \
  --movement-ordered-rays \
  --batch-size 256 --epoch-size 1000000 --max-epochs 20 --seed 42 \
  --default-root-dir runs/ray-ordered

python -m pytest tests/test_movement_network.py tests/test_ordered_rays.py -q
```

Use game-disjoint training and validation files, repeat seeds, and compare both
equal-position budgets and equal training time. Report held-out loss, tactical
subsets and batch-one CPU inference time. Training on the bundled small.binpack
is only a pipeline smoke test, not evidence of generalization or playing strength.
The native loader can introduce sampling variation even with the same Torch
seed. A controlled comparison should additionally fix the data stream.

Ordered models save as training checkpoints or `.pt` models. Both existing
Stockfish export formats are guarded: `.nnue` cannot represent movement graphs,
and version-2 `.mnnue` has no channel-mixing runtime. Native execution, parity
checks and matched-time engine games are still required before claiming an Elo
gain. This change implements the experiment in the training/test suite, not a
new native Stockfish evaluator.

## Further changes to the supplied design

* Separate piece and square states is a larger, independent ablation. Establish
  whether this compact order-sensitive scan helps before adding six wide blocks.
* Gated sum is still a sum; it does not inherently solve information loss.
  Test separate friendly/enemy aggregation and preserve cardinality before
  introducing per-square attention.
* Horizontal reflection is not a general standard-chess symmetry when castling
  rights remain. Standard castling destinations are not reflected onto their
  counterparts. Disable that augmentation for these positions or explicitly
  model the correct rules. Rank reflection plus color swap is a safer symmetry.
* HalfKA decoding loses castling rights, en passant and history. Therefore this
  existing training interface cannot implement the proposal's full global
  state. Extending the loader is a separate necessary step if targets depend on
  that information. Do not claim exact legal-position symmetry from board-only
  encodings.
* Benchmark the complete graph extraction and evaluator inside search. Fewer
  edges than dense attention does not demonstrate NNUE-like incremental cost.

## Tests

Tests cover exact baseline equivalence at initialization, nonzero mixer
gradients, distinguishable path order, linear source superposition, the
transition norm bound, incremental/full parity after several board edits,
optimization through the sparse training interface, checkpoint restoration,
and rejection by the incompatible native exporter.
