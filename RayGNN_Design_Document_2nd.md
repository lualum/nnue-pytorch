# RayGNN Model Implementation Specification

## Part I. Updated design

### 1. Fixed configuration

Implement this configuration as the default.

| Component | Specification |
|---|---|
| Nodes | All 64 squares, including empty squares |
| Directed candidate edges | 1,792 |
| Graph layers | 2, with separate parameters |
| Node width | 64 |
| Message width | 32 |
| State width | 16 |
| Piece embedding width | 16 |
| Node update MLP | 144 → 64 → 64 |
| Per-square readout MLP | 128 → 128 → 128, SiLU after both layers |
| Global pooled width | 128 |
| Raw-board bypass width | 832 |
| Value head | 976 → 64 → 1 |
| Hidden activation | SiLU |
| Learned residual scale | One scalar per graph layer, initialized to 0.1 |

Share parameters across squares and edges within each layer. Reuse one edge encoder across both layers. Use synchronous node updates, scalar message/feedback gates, and global mean pooling. Do not add spatial pooling regions, piece-type pooling groups, fixed material values, king-relative inputs, layer normalization, dropout, attention softmax or internal search.

### 2. Inputs and encoding

Index squares as `square = 8 * rank + file`, with `a1 = 0`, `h8 = 63`. Files and ranks run from 0 through 7 in White's orientation.

| Input | Shape | Encoding |
|---|---|---|
| `piece` | `[B,64]` | 0 empty; 1–6 White P/N/B/R/Q/K; 7–12 Black P/N/B/R/Q/K |
| `side_to_move` | `[B,1]` | +1 White, -1 Black |
| `castling` | `[B,4]` | Binary WK/WQ/BK/BQ |
| `en_passant` | `[B]` | Square 0–63, or 64 for none |

Construct a 70-dimensional state input from side to move, four castling bits and the 65-dimensional en-passant one-hot encoding:

```
s = Linear(70,16)(state_input)
coords_i = [file_i / 7, rank_i / 7]
h_i^0 = Linear(18,64)(concat(Embedding(13,16)(piece_i), coords_i))
```

Use the en-passant target recorded by the position format consistently in training and inference. Legal-move handling remains outside the network.

If training labels depend on halfmove or repetition state, append the available draw-state fields to `state_input` and adjust the state encoder's input width. Specify their encoding in the model configuration and checkpoint. Do not mix datasets with incompatible state contracts. Terminal and history-dependent draw adjudication remains in the engine.

Retain `h^0` and the original `piece` tensor for the final readout.

### 3. Static graph

Create an edge `i -> j` for each distinct pair satisfying either condition:

- Same rank, file or diagonal: 1,456 directed edges.
- Knight displacement `(abs(df), abs(dr))` equal to `(1,2)` or `(2,1)`: 336 directed edges.

Include both directions, all ray distances, empty endpoints and blocked pairs. Do not add self-edges or duplicate pawn/king edges.

Register these tensors as model buffers:

```
src, dst:       [1792]       integer endpoint indices
between:        [1792,6]     intervening squares, ordered from source
between_mask:   [1792,6]     valid intervening slots
relative_delta: [1792,2]     signed target-minus-source file/rank offsets
is_knight:      [1792]       binary relation type
```

Use a safe square index for padded `between` slots and mask their gathered occupancy before any reduction. Knight edges have no intervening squares.

### 4. Dynamic edge features

For each edge, construct this 31-dimensional vector:

| Feature | Width | Encoding |
|---|---:|---|
| Relative displacement | 2 | `[df/7, dr/7]` |
| Knight relation | 1 | Binary |
| Intervening blocker count | 7 | One-hot 0 through 6 |
| First blocker piece | 13 | One-hot piece ID; 0 means no blocker |
| First blocker displacement | 2 | Relative to source, divided by 7; zeros if absent |
| Blocker present | 1 | Binary |
| Source attack geometry matches | 1 | Binary, ignoring intervening occupancy |
| Direct attack/defense | 1 | Binary, including intervening occupancy |
| Source occupied | 1 | Binary |
| Destination occupied | 1 | Binary |
| Pawn forward geometry | 1 | Binary, ignoring occupancy |

`Source attack geometry matches` is true for compatible rook/bishop/queen rays, knight jumps, one-square king adjacency, or one-square pawn diagonals in the pawn's forward direction. It is false for empty sources.

`Direct attack/defense = geometry_matches AND blocker_count == 0`. Friendly-occupied destinations count as defended squares. Pins and legal king-move restrictions do not alter this flag. Destination occupancy does not count as an intervening blocker.

`Pawn forward geometry` is true for a one-square forward displacement, or a two-square forward displacement from the pawn's starting rank. It is not an attack or legality flag.

Compute the edge embedding once per position and reuse it across graph layers:

```
e_ij = Linear(31,32)(edge_features_ij)
```

All candidate edges carry contextual messages regardless of their attack flag or blocker count. Empty sources may send contextual messages but always have false attack flags.


### 5. Graph layer

For each layer, project source and target states once per node:

```text
p = A_l(h)                  # Linear(64,32), no bias
q = B_l(h)                  # Linear(64,32), no bias
c = C_l(s)                  # Linear(16,32), no bias

x_ij = SiLU(p_i + q_j + e_ij + c)
message_ij  = sigmoid(a_l dot x_ij + a0_l) * x_ij
feedback_ij = sigmoid(b_l dot x_ij + b0_l) * x_ij
```

Each gate has a learned 32-vector and scalar bias. Gather source and target projections using the static edge indices. A source's feedback depends on both its own state and the target's state.

Aggregate all messages into their destinations and feedback into their sources:

```text
incoming = zeros(B,64,32)
outgoing = zeros(B,64,32)
incoming.scatter_add_(destination_index, message)
outgoing.scatter_add_(source_index, feedback)

v_i = concat(h_i, incoming_i, outgoing_i, s)  # 144
u_i = Linear(64,64)(SiLU(Linear(144,64)(v_i)))
h_i_next = h_i + alpha_l * u_i
```

Broadcast state features across nodes. Do not divide incoming/outgoing sums by degree. Compute every edge from the same previous-layer node tensor, finish both reductions, then update all nodes together. Repeat for the second layer using its own parameters.

### 6. Global readout

Apply one shared nonlinear readout to each square:

```text
a_i = concat(h_i^2, h_i^0)                 # 128
b_i = SiLU(Linear(128,128)(a_i))
t_i = SiLU(Linear(128,128)(b_i))           # 128
z = sum(t_i, square_dimension) / 64       # [B,128]
```

Pool all 64 squares, retaining empty-square contributions. Coordinates remain available through the original node state. Use no spatial partitions or piece-specific pooling masks.

### 7. Raw-board bypass and value head

```text
r = flatten(one_hot(piece, num_classes=13))  # [B,832]
v = concat(z, r, s)                         # [B,976]
V_white = Linear(64,1)(SiLU(Linear(976,64)(v)))
```

Flatten in square-major order. Output an unbounded White-positive scalar in pawn units. Do not add a fixed material correction or clip the neural output inside the model.

### 8. Forward contract and shapes

```text
forward(piece, side_to_move, castling, en_passant, draw_state=None)
    -> V_white  # [B,1]
```

| Tensor | Shape |
|---|---|
| Initial and updated nodes | `[B,64,64]` |
| Encoded state | `[B,16]` |
| Edge features | `[B,1792,31]` |
| Edge embeddings, messages, feedback | Each `[B,1792,32]` |
| Incoming and outgoing sums | Each `[B,64,32]` |
| Node update input | `[B,64,144]` |
| Per-square readout | `[B,64,128]` |
| Global pooled vector | `[B,128]` |
| Raw-board bypass | `[B,832]` |
| Head input | `[B,976]` |
| Value | `[B,1]` |

Linear layers have biases unless explicitly disabled. Use framework-default linear/embedding initialization, zero gate biases, and residual scales initialized to 0.1. Store architecture dimensions, feature ordering, perspective and input schema in checkpoints. Retain integer indices and match floating feature dtypes to the relevant projections.

At the engine boundary:

```text
V_side_to_move = side_to_move * V_white
score_cp = round(100 * V_side_to_move)
```

Clamp converted scores below the engine's reserved mate-score range. Search owns legal moves, terminal positions and draw adjudication. The reference implementation recomputes the full graph for each evaluated position.

## Part II. Optional changes and training allocation

### 9. Shared run budget

The allocations below are practical starting budgets, not convergence guarantees. Use one **fixed-size training epoch of 1,000,000 sampled positions** for this plan. For a streaming dataset, this means a training interval, not a full dataset pass. If retaining another epoch size E, multiply every epoch count below by `1,000,000 / E`, rounding up. Count positions across all GPUs together.

Use seeds **17, 29 and 43**. Screen each architecture with seed 17 for **20 epochs**. For a promising candidate, continue seed 17 to **50 total epochs**, and train seeds 29 and 43 for **50 epochs each**. Confirmation therefore means **50 epochs × 3 seeds total**, including the initial screening run. Run the reference under the same budget. If both comparison arms are still improving at the limit, extend both by **10 epochs per seed** together.

For each run, configure the learning-rate schedule for the 50-epoch horizon from the beginning; screening is a checkpoint at epoch 20, not a separate schedule. Keep data order paired by seed, effective batch size, loss, optimizer, augmentation and target scaling identical across architecture comparisons. Initialize each changed architecture from scratch. Resume only its own screening checkpoint during confirmation.

Use a fixed held-out split to select candidates. Split by game where available and remove duplicate positions across splits. Retain the established target transform and loss for comparisons. For a fresh teacher-score pipeline, use White-positive `cp / 100` targets and mean Huber loss with delta 1.0, AdamW and gradient-norm clipping at 1.0; exclude mate labels unless separately encoded.

Complete the reference and original-design comparison first. Screen optional changes one at a time in the order below, then confirm only the strongest one or two. Compare any final combination against the reference using the same 50-epoch, three-seed budget. Include inference cost when selecting the final model.

### 10. Architecture changes

All changes below are disabled in Part I. Budgets apply separately to each listed variant.

| Priority | Change | Exact implementation | Screen | Confirmation |
|---|---|---|---|---|
| Reference | Part I design | Two layers, 32-wide messages, nonlinear 128-wide readout | 20 epochs × 1 seed | 50 total epochs × 3 seeds |
| Control | Original compact design | M=16; update input 112; readout `128 → 64 → 64` with SiLU only between its layers; pooled width 64; head `912 → 64 → 1` | 20 × 1 | 50 × 3 |
| Control | No message passing | Keep Part I readout/head; feed `concat(h0,h0)` to readout and omit graph layers | 20 × 1 | 50 × 3 if needed to resolve graph benefit |
| Control | Raw-board-only | Board plus state; head `848 → 64 → 1`; omit node encoder and graph/readout branch | 20 × 1 | 50 × 3 if competitive |
| 1 | Channel-wise gates | Replace each scalar gate with `sigmoid(Linear(32,32)(x))`; retain elementwise multiplication by x; use separate message and feedback gate maps | 20 × 1 | 50 × 3 if promising |
| 2 | King-relative coordinates | Append four offsets to initial node input; details below | 20 × 1 | 50 × 3 if promising |
| 3 | Smaller messages | M=16; update MLP `112 → 64 → 64`; retain Part I readout | 20 × 1 | 50 × 3 if competitive at lower cost |
| 4 | Smaller readout | Readout `128 → 64 → 64` with SiLU after both layers; head `912 → 64 → 1` | 20 × 1 | 50 × 3 if competitive at lower cost |
| 5 | Relation-specific gate biases | Three edge classes with separate scalar biases for message and feedback gates | 20 × 1 | 50 × 3 if promising |
| 6 | One graph layer | L=1; readout uses `concat(h1,h0)` | 20 × 1 | 50 × 3 if competitive at lower cost |
| 7 | Three graph layers | L=3; readout uses `concat(h3,h0)` | 20 × 1 | 50 × 3 if promising |

For channel-wise gates, initialize gate weights from the framework default and biases to zero. Message and feedback can then emphasize different coordinates of the shared pair vector. Do not add a bias-free linear map solely before each sum: it can be moved after the sum and absorbed into the following linear update, so it is not a meaningful expressiveness upgrade by itself.

For king-relative coordinates, append file/rank displacement from the White king, then file/rank displacement from the Black king, all divided by 7. Initial node encoder input changes from 18 to 22 dimensions. Keep absolute coordinates and the same 64-dimensional node output. Require exactly one king of each color. Do not add king-centered pooling.

For relation-specific biases, assign each edge exactly one class in this order: direct attack/defense; occupied source without direct attack/defense; empty source. Add a learned class bias to each existing scalar gate logit, with separate bias vectors per layer and gate. Initialize the biases to zero and retain all edges.

### 11. Training and implementation changes

| Change | Implementation | Epoch/seed allocation |
|---|---|---|
| Color symmetry augmentation | With probability 0.5, reflect ranks and swap colors; transform state and negate the White-positive target | 20 epochs × 1 seed; 50 total × 3 if promising, compared against identical unaugmented runs |
| Float32 reductions with mixed precision | Accumulate incoming/outgoing sums in float32, then cast as needed for node updates | 1 epoch × 1 seed for numerical checks; no architecture sweep |
| Edge chunking/checkpointing | Process fixed edge chunks; finish reductions before any node update; checkpoint edge computation if backward memory remains excessive | 1 epoch × 1 seed for forward/backward equivalence and memory checks |
| Sparse raw-board projection | Implement the raw-board block of the first head affine map as 64 indexed weight-row lookups and a sum; retain graph/state affine contributions and one bias | 0 training epochs, 0 seeds; exact-weight output-equivalence checks |
| Compiled/fused graph operations | Fuse gathers, pair activation, gates and reductions without changing equations | 0 training epochs, 0 seeds; compare forward outputs and gradients against the reference |
| Quantization-aware fine-tuning | Start from each selected trained checkpoint; calibrate activation ranges and fine-tune the selected quantized implementation | 5 additional epochs × 3 checkpoint seeds; compare with 5-epoch float continuations |

For rank reflection, keep file fixed and set `rank -> 7-rank`; swap piece colors and side to move, exchange WK/BK and WQ/BQ rights, reflect the en-passant square, and negate the target. Keep clocks unchanged and transform encoded history consistently. Do not substitute a 180-degree rotation or unconditional file reflection when castling rights remain.

Edge chunking alone does not guarantee a reduction in all activations retained for backward. Preserve the same logical model and verify chunked versus unchunked computation before using it for training. Implementation-only changes do not require fresh multi-seed architecture experiments unless they change numerical behavior materially.
