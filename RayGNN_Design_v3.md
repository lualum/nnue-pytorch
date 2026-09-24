# RayGNN v3: standalone relational evaluator

## Part I. Default implementation

### 1. Scope and fixed configuration

Build one jointly trained, standalone network that maps a chess position to a White-positive evaluation in pawn units. No external evaluator runs at inference. Train all components from scratch. Search, legality, terminal adjudication, and history-dependent draws remain engine responsibilities.

This is the recommended next candidate, not a demonstrated improvement. Its accuracy and inference cost must pass the gates in Part II before a long training run. The previous checkpoint is a comparison model, not a compatible initialization.

| Component | Default |
|---|---|
| Nodes | All 64 squares, including empty squares |
| Node width D | 96 |
| State width S | 16 |
| Graph layers L | 2, separate parameters |
| Relationship channels R | 3: direct, x-ray, context |
| Message width M | 32 per channel per direction |
| Pair interaction | 128 → 64 → 64, last output split into incoming/outgoing messages |
| Node update | 310 → 192 → 96 |
| Residual scale | Learned scalar per layer, initialized to 0.1 |
| Local contextual readout | 192 → 64 → 1 |
| Spatial readout | Shared 192 → 8 per square, flatten to 512 |
| Global head | 528 → 128 → 1 |
| Activations | SiLU in hidden layers |
| Normalization | Pre-LayerNorm on graph node states; square-root count scaling on aggregates |
| Output | Learned piece-square sum + contextual square sum + spatial global head |

Do not add attention softmax, dropout, internal search, fixed material values, or a separately trained base evaluator. Do not replace the spatial readout with mean pooling in the default.

### 2. Input contract

Index squares as `8 * rank + file`, with a1 = 0 and h8 = 63, in White's orientation.

| Input | Shape | Encoding |
|---|---|---|
| piece | [B,64] | 0 empty; 1–6 White P/N/B/R/Q/K; 7–12 Black P/N/B/R/Q/K |
| side_to_move | [B,1] | +1 White, -1 Black |
| castling | [B,4] | WK, WQ, BK, BQ binary flags |
| en_passant | [B] | Target square 0–63; 64 for absent |

Encode state as the concatenation of side to move, castling, and a 65-way en-passant one-hot vector:

```text
state_input: [B,70]
s = Linear(70,16)(state_input)
```

Use the en-passant target recorded by the dataset consistently. If labels require available halfmove/repetition fields, extend the state input explicitly and store that schema in the checkpoint. Do not silently reconstruct missing history.

Targets are White-positive `centipawns / 100`. Convert side-to-move-positive source scores exactly once. Exclude mate sentinels unless using a separately specified target encoding. Preserve unbounded predictions during loss calculation.

### 3. Initial node representation

Use separate learned piece and joint piece-square tables:

```text
joint_id_i = 64 * piece_i + square_i
h0_i = E_piece[piece_i] + E_joint[joint_id_i]

E_piece: [13,96]
E_joint: [832,96]
```

Initialize both embedding tables with independent Normal(0, 0.05) entries. Include empty-square entries. Retain `h0` unchanged for the final readout. Absolute location is represented directly by the joint table; no additional coordinate projection is needed in the default. Relative geometry remains in edge features.

Do not add king-relative embeddings initially. Test them after the core model passes its learning and cost gates.

### 4. Candidate geometry and active relationships

Precompute all directed same-rank/file/diagonal pairs and knight-displacement pairs, excluding self-edges and duplicates. There are 1,456 ray pairs and 336 knight pairs, for 1,792 candidates. Both directions exist as candidates; their active classes can differ.

Register source/destination indices, ordered intervening square indices, valid-slot masks, signed displacement, and a knight flag as model buffers. Ray pairs have at most six intervening squares. Padded indices must be valid and their gathered occupancy masked out.

For each candidate `i → j`, compute occupancy and attack geometry from the original board. Endpoint j does not count as an intervening blocker. Classify in the following precedence order; retain at most one class per directed pair:

| Class | Activation rule |
|---|---|
| Direct | Occupied source; its attack geometry matches the displacement; zero intervening blockers |
| X-ray | Occupied bishop/rook/queen; compatible slider direction; exactly one intervening blocker |
| Context | Not direct or x-ray, and either king-adjacent displacement for any source, or pawn-forward geometry for an occupied pawn |
| Inactive | All remaining candidates |

Attack geometry means rook/bishop/queen rays, knight jumps, king adjacency, and pawn diagonals in the source pawn's forward direction. Friendly-occupied targets count as defended. Empty targets remain included. Pins and legal king safety do not change the direct flag.

Pawn-forward geometry means one step forward, or two steps from the starting rank, regardless of occupancy. It supplies contextual information, not move legality. The context class includes empty-source adjacency, allowing empty-square states to participate.

The x-ray class includes empty targets beyond the first blocker and the next occupied target, but excludes targets with two or more intervening blockers. It does not assert that the blocker can legally move or that the relationship is a pin.

Connectivity is computed once per board and reused for both graph layers. Never update the board or edge masks from learned node states. All active edges in a layer are evaluated synchronously.

### 5. Edge representation

For each retained directed edge, construct the following 31 features:

| Feature | Width |
|---|---:|
| Signed file/rank displacement divided by 7 | 2 |
| Knight-displacement flag | 1 |
| Intervening blocker count, one-hot 0–6 | 7 |
| First blocker piece ID, one-hot 0–12; 0 if absent | 13 |
| First blocker file/rank displacement from source divided by 7; zero if absent | 2 |
| Blocker-present flag | 1 |
| Source attack geometry matches | 1 |
| Direct attack/defense flag | 1 |
| Source occupied | 1 |
| Target occupied | 1 |
| Pawn-forward geometry flag | 1 |

```text
e_ij = Linear(31,32)(edge_features_ij)
```

Share this edge encoder across layers and relationship classes. Compute its output once per position. Relationship classes receive separate pair-interaction MLPs below.

### 6. Graph layer

For each layer l, apply nodewise LayerNorm over 96 features, with epsilon 1e-5 and learned affine parameters. The residual stream itself is not normalized in place.

```text
n = LayerNorm_l(h)
p = A_l(n) + C_l(s)       # [B,64,32], broadcast state
q = B_l(n)                # [B,64,32]
```

`A_l` and `B_l` are bias-free 96 → 32 maps. `C_l` is a bias-free 16 → 32 map. Compute them once per node, not once per edge.

For an active edge of class r:

```text
a_ij = concat(p_i, q_j, p_i * q_j, e_ij)  # 128
w_ij = Linear_l,r(64,64)(SiLU(Linear_l,r(128,64)(a_ij)))
message_ij, feedback_ij = split(w_ij, [32,32])
```

Use independent pair MLP parameters for all three classes and both layers. The multiplicative term is elementwise. Incoming and outgoing messages are separately learned outputs, not scalar multiples of the same vector. Do not add sigmoid gates in the default.

Accumulate in float32 separately by node, class, and direction:

```text
I_i,r = sum(message_ji for active edges j → i of class r)
O_i,r = sum(feedback_ij for active edges i → j of class r)
dI_i,r = number of active incoming edges of class r
dO_i,r = number of active outgoing edges of class r

I_i,r = I_i,r / sqrt(max(1, dI_i,r))
O_i,r = O_i,r / sqrt(max(1, dO_i,r))
c_i = concat(log1p(dI_i,r), log1p(dO_i,r) for r in fixed class order)
```

There are six count features. Use natural logarithms. A zero-degree aggregate is exactly zero. Cast aggregates to the update computation dtype after float32 reduction and scaling.

Concatenate in this exact order:

```text
v_i = concat(
    n_i,                            # 96
    I_direct, I_xray, I_context,     # 96
    O_direct, O_xray, O_context,     # 96
    c_i,                            # 6: incoming three, outgoing three
    s                               # 16
)                                   # total 310

u_i = Linear_l(192,96)(SiLU(Linear_l(310,192)(v_i)))
h_next_i = h_i + alpha_l * u_i
```

Initialize `alpha_l = 0.1`. Do not detach messages, node states, aggregates, or counts from their intended computation; integer topology/counts themselves require no gradients. Complete every aggregate from the old `h` before assigning `h_next`. Repeat for the second layer using separate parameters.

### 7. Unified output

All output components are optimized together using one loss. There is no pretrained component or fixed material score.

**Learned linear piece-square contribution**

```text
V_linear = sum_i T[piece_i, square_i]
```

`T` has 12 × 64 learned scalar entries for occupied pieces, initialized to zero. Empty-square contributions are fixed to zero and have no learned table entries. The table learns signed values; do not hardcode color antisymmetry or material constants. It is a direct output path within this network.

**Contextual square contribution**

```text
a_i = concat(h2_i, h0_i)                  # 192
local_i = Linear(64,1)(SiLU(Linear(192,64)(a_i)))
V_local = sum_i local_i / 8
```

Apply the same local MLP to all squares, including empty squares. The divisor is a fixed initialization-scale choice, not a material normalization. Do not divide by occupied-piece count. These contributions depend on other pieces through message passing.

**Position-preserving global contribution**

```text
b_i = SiLU(Linear(192,8)(a_i))            # [B,64,8]
b = flatten_square_major(b_i)             # [B,512]
V_global = Linear(128,1)(SiLU(Linear(528,128)(concat(b,s))))

V_white = V_linear + V_local + V_global    # [B,1]
```

Use a shared projection to eight channels per square, then flatten in a1 through h8 order. The global head has location-specific weights and can combine arbitrary square pairs through its hidden units. There are no manually selected pooling regions, piece-type pools, or color pools.

Initialize the final weights of the local and global heads with Normal(0, 0.01), and their final biases to zero. Use ordinary linear-layer initialization elsewhere. Both contextual heads receive gradients from the start. Do not separately pretrain the linear table.

### 8. Tensor and inference contract

| Tensor | Shape |
|---|---|
| h0, h1, h2 | [B,64,96] |
| s | [B,16] |
| Packed edge features | [E_active,31] across the batch |
| Packed edge embeddings | [E_active,32] |
| Pair input | [E_active,128] |
| Incoming/outgoing aggregates | Each [B,64,3,32] |
| Incoming/outgoing counts | Each [B,64,3] |
| Node update input | [B,64,310] |
| Readout input | [B,64,192] |
| Flattened spatial features | [B,512] |
| Final prediction and target | Both [B,1] |

```text
forward(piece, side_to_move, castling, en_passant, draw_state=None)
    -> V_white

engine_score_cp = round(100 * side_to_move * V_white)
```

Clamp only the converted engine score below its reserved mate range. Store input schema, class ordering, architecture version, feature order, output perspective, and target transform in the checkpoint.

### 9. Compute implementation

Use a masked dense reference implementation for correctness, then a packed active-edge implementation for cost measurement. In the packed path, collect `(batch, source, target, class)` and run each class MLP only on its retained edges. Flatten node indices as `batch * 64 + square` for reductions. Never mix examples during scatter operations. Handle an empty class without special learned outputs.

Computing every candidate MLP and masking afterward does not deliver the proposed sparsity savings. Packing, dynamic shapes, and reductions also incur overhead, so measure end-to-end performance rather than counting retained edges alone.

Reuse edge features across layers. Avoid materializing `[B,1792,96]` source and target tensors when packing is available. Chunk active-edge pair computations if needed; finish all sums before updating nodes. Chunking without activation checkpointing may not reduce retained backward memory substantially.

Log active-edge counts by class, parameters, peak training memory, training positions/second, batch-one CPU latency, and batched GPU throughput. Include feature construction in inference timing. The reference network recomputes the graph for each position; incremental updates and quantization are later projects.

## Part II. Verification, experiments, and training allocation

### 10. Correctness gates before architecture training

1. Decode and inspect at least 20 positions. Verify piece IDs, a1/h8 indexing, pawn direction, castling, en-passant, target units, and score perspective. Assert prediction and target shapes are identical before loss calculation.
2. Check graph fixtures covering friendly defense, empty attacked squares, a single blocker, two blockers, knight jumps, pawn advances, and board boundaries. Verify relation precedence and zero-degree behavior.
3. Compare dense and packed implementations at identical weights: outputs, loss, and gradients. Use float32 and tolerances appropriate to accumulation order. No cross-example edges are permitted.
4. Overfit 256 fixed examples with no conflicting labels, no augmentation, no weight decay, and full float32. Try learning rates 1e-4, 3e-4, and 1e-3 for up to 2,000 updates each. Seek training Huber below 0.01; this is a diagnostic target, not a theorem. Failure blocks a large run until explained.
5. Fit synthetic signed material labels on diverse boards and evaluate on unseen boards. The direct learned table must make this task easy. Separately train with the table disabled on synthetic attack-count labels to verify that the graph path learns and receives gradients.
6. Log actual learning rate, unclipped gradient norm, clipping frequency, skipped AMP updates, residual scales, parameter changes, and per-branch gradient norms. Verify resumed optimizer/scheduler state rather than assuming it is restored.
7. For distributed runs, inspect example hashes across ranks, shard streams, count global examples correctly, and compare reduced loss against a direct calculation. Synchronize optimizer steps. Do not call repeated rank-identical batches distinct training data.

These checks need one seed, 17. They are not full training epochs and do not require three-seed replication.

### 11. Data and training protocol

Build a fixed 100,000-position validation set disjoint from training, split by game where possible and deduplicated by the full encoded position. If only one binpack exists, partition it or filter reserved examples out of the stream; do not evaluate a holdout that is also used for training. Retain an untouched final test set for the selected model.

Use mean Huber loss with delta 1.0 on pawn-valued targets. Also report MAE, median absolute error, and metrics in absolute-target buckets [0,1), [1,3), [3,8), and [8,infinity). Huber is not MAE. Preserve identical target transforms and filtering for all model comparisons, including NNUE controls.

| Training setting | Starting value |
|---|---|
| Optimizer | AdamW, betas (0.9,0.999), epsilon 1e-8 |
| Peak learning rate | 3e-4 |
| Effective global batch | 1,024 positions; microbatch and accumulation as needed |
| Weight decay | 1e-4 for dense matrices and vector embedding tables |
| No-decay parameters | Biases, LayerNorm parameters, residual scales, scalar piece-square table |
| Warmup | First 200 optimizer updates |
| Schedule | Cosine from peak to 3e-5 over a predeclared 20-million-position horizon |
| Gradient clipping | Global norm 1.0, after AMP unscaling |
| Precision | Float32 correctness baseline; AMP with float32 reductions after equivalence checks |
| Validation frequency | Every 250,000 global positions, approximately |
| Checkpoints | Last and best fixed-validation loss, with optimizer/scheduler/scaler/RNG state |

Use 50% color-symmetry augmentation in production candidates: reflect ranks (`rank -> 7-rank`), swap colors and side to move, exchange White/Black castling rights on the same wing, reflect en-passant, and negate White-positive targets. Keep clocks unchanged and transform any encoded history consistently. Apply the same augmentation policy to all comparison arms. Do not use unconditional file reflection with castling rights.

Log examples, unique-example estimates, optimizer steps, elapsed time, and learning rate together. Keep effective batch size fixed across GPU counts. If training proceeds beyond the declared schedule, predeclare a continuation schedule and apply it equally to comparison arms; do not restart warmup accidentally.

### 12. Required controls and budgets

One epoch below means 1,000,000 globally sampled training positions, not one dataset pass. Use seed 17 for screening. All new configurations start from scratch. Small tests are intended to reject gross failures; early rankings can change.

| Run | Exact difference | Initial screen | Extension |
|---|---|---|---|
| Old default | Previous specification, same corrected pipeline | 5 epochs × 1 seed; existing checkpoint also evaluated separately | Only if needed to establish a fair baseline |
| v3 default | All of Part I | 5 epochs × 1 seed | 20 total epochs if promising |
| No message passing | Omit graph layers; readout receives concat(h0,h0); retain all output terms | 5 × 1 | 20 total if competitive |
| No relation separation | Same active edges, one shared pair MLP; aggregate into one incoming and one outgoing channel; update input 178 → 192 → 96 | 5 × 1 | 20 total if competitive |
| Dense contextual edges | Restore all inactive candidates as context; retain direct/x-ray precedence and v3 equations | 5 × 1 | 20 total only if added accuracy warrants its cost |
| NNUE control | Standalone comparator only; identical data/loss evaluation | 5 × 1 | Match finalist budgets where feasible |

For the no-relation-separation control, concatenate 96 node features, 32 incoming, 32 outgoing, two log-counts, and 16 state features. It deliberately reduces parameters as well as changing routing; report that difference. It is a practical control, not a parameter-matched causal proof.

At 1 million positions, reject only clear failures such as negligible learning, unstable updates, or near-constant predictions. At 5 million, compare fixed validation and wall-clock curves. Continue candidates that are competitive in error or show a credible accuracy/cost advantage. A 10% relative Huber improvement at matched examples is a useful provisional definition of significant improvement, not a statistical guarantee or an engine-strength criterion.

Confirm the best one or two candidates at 20 million positions using seeds 17, 29, and 43. Continue seed 17 from its screening checkpoint; train the other seeds from scratch. If still improving materially, extend the finalists and their controls together to 50 million, with a declared continuation schedule. Do not allocate 50 million × 3 to every variant.

### 13. Follow-up changes, in order

Run these only after the default passes correctness and shows useful learning. Change one item at a time. Each receives 5 epochs × seed 17; extend promising candidates to 20 total epochs × seeds 17, 29, 43. Keep data order, batch size, target, loss, and schedule paired.

| Priority | Variant | Exact implementation |
|---|---|---|
| 1 | King-relative information | Append four file/rank offsets to the two kings, divided by 7, to a bias-free 4 → 96 projection added to h0. Require one king per color. |
| 2 | Wider spatial readout | Project each square 192 → 16; flatten to 1,024; global head 1,040 → 128 → 1. |
| 3 | Wider messages | M=48; pair input 192 → 96 → 96; node update input 406 → 192 → 96. |
| 4 | Global pooling alternative | Shared 192 → 128 → 128 SiLU readout, mean over 64 squares; global head 144 → 128 → 1. Retain linear/local terms. |
| 5 | Third graph layer | Separate third layer with identical dimensions; readout uses h3 and h0. |
| 6 | Simpler pair interaction | Remove p*q; pair input becomes 96 → 64 → 64. |

Use branch interventions on trained checkpoints to inspect utilization: zero individual output terms, replace graph changes with zero (`hL = h0`), and inspect sensitivity to color/position transformations. These interventions change the input distribution and cannot replace retrained ablations.

Do not change the target transform just to lower the reported number. A future bounded-outcome objective is a separate experiment requiring both models to be evaluated under the same external metrics.

### 14. Selection and engine integration gate

Select using a curve of held-out error versus actual inference latency, not parameter count or training loss alone. Compare training speed separately from inference speed. Report batch-one CPU timing for alpha-beta integration and batched GPU timing only if the intended engine actually batches evaluation.

If the graph does not beat the no-message control at an acceptable cost, stop increasing its training budget and revise the interaction design. If gains are confined to extreme score buckets while near-equal positions worsen, investigate before accepting the model.

After selecting a trained model, cross-check training and engine outputs on fixed positions before playing matches. Test paired openings with colors reversed and equal hardware/time limits. Report uncertainty rather than treating a small match score as conclusive. Lower teacher-score error alone does not establish stronger play.

The desired result is a standalone evaluator whose learned relationships justify their inference cost. No performance claim is established by this specification alone.
