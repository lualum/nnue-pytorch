# RayGNN: A Ray-Based Graph Neural Network for Chess Evaluation

**Status:** Proposed research architecture, v0.1  
**Primary use:** Position evaluation within an alpha-beta search engine  
**Research question:** Can simultaneous chess-specific message passing improve the quality/compute trade-off between compact NNUE and larger neural evaluators?

## 1. Scope and success criteria

RayGNN treats each of the 64 squares as a node. It passes messages along eight queen-like directions, plus separate knight and pawn relations, then produces a scalar evaluation. All node updates within a layer use the same previous-layer state, not sequential board scanning.

The initial version prioritizes a correct, testable architecture over NNUE-style incremental inference. Success requires improvements at matched training data and inference budgets, and ultimately at fixed search conditions. Validation loss alone is insufficient.

**Non-goals for v0.1:** policy prediction, move generation, fully dynamic arbitrary graphs, incremental accumulator updates, and proof of superior playing strength.

## 2. Input contract and information preservation

Inputs: `piece[64]` (13 categorical values: empty and 12 colored piece types), side to move, castling rights, legal en-passant target or none, and optionally halfmove clock/repetition information if the target score depends on draw rules. The position encoding must be lossless for the state relevant to the chosen target. An evaluation head cannot recover missing castling or en-passant rights from piece placement.

Encode each square with a learned piece embedding, learned or deterministic square coordinates, king-relative offsets for both kings, and compact state features. Concatenate raw features and project to `d_model=64`. Keep the original piece/square features available to the final readout. The initial feature dimensions are design choices, not claims that each group must occupy an exact number of dimensions.

**Perspective:** use either fixed White-oriented coordinates and a White-positive scalar, or canonicalize to side-to-move perspective. Choose one convention end to end. This specification uses fixed coordinates and White-positive evaluation. Black/White color-swap plus 180-degree board rotation should negate the output when all applicable state fields are transformed consistently.

## 3. Geometry and message semantics

Precompute the eight ordered rays for every square. A ray consists of up to seven squares, with direction and distance. For each source/destination pair, distinguish:

- Geometric alignment: squares share a rank, file, or diagonal, independent of occupancy.
- Direct visibility: no occupied square lies strictly between source and destination.
- First blocker: identity, color, location, and distance of the nearest intervening piece, if any.
- X-ray relation: destination is beyond one or more blockers; represent at least the first blocker and preferably a clipped blocker count.
- Actual attack: source piece attacks the destination under chess attack semantics. Rooks/bishops/queens follow compatible unobstructed rays; pawns attack diagonally by color; knights use separate jump edges. King adjacency is another local relation. Distinguish an attacked square from a legal move, since pins and king safety affect legal moves.

**Important:** an empty square can be attacked; a piece on the destination does not count as an *intervening* blocker. A ray is not automatically an attack. Geometric edges from empty source squares should be masked for piece-generated messages, while empty destination nodes may receive messages. Pawn forward movement is not a pawn attack.

For efficiency, store geometric ray indices as fixed tensors. Compute occupancy-dependent masks and first-blocker features per position, vectorized across batch, squares, directions and distance. Initially allow messages to all geometrically aligned destinations with separate direct and x-ray masks. Compare this with a first-visible-piece-only sparse variant.

## 4. Message-passing block

**Recommended initial configuration:** `d_model=64`, `d_msg=16` per queen direction, three layers. Each layer computes all messages from the previous layer's square states in parallel.

For destination `i`, source `j`, direction `r`, and layer `l`:

`m[l,r,j->i] = MLP_r([h[l,j], h[l,i], edge(j,i)])`

Use shared parameters by direction class (orthogonal versus diagonal) plus a learned direction embedding, rather than eight unrelated large MLPs in v0.1. Edge features include relative distance, direct/x-ray flags, first blocker features and source/destination piece categories. A small gated message or masked weighted sum is the baseline; do not assume full attention is required.

Aggregate *within each direction* using a masked weighted sum or a sum plus count normalization. Preserve direct and x-ray information with distinct edge features or separate channels. An unrestricted mean may erase multiplicity, so include message counts and/or summed messages. An unrestricted sum may make outputs sensitive to the number of geometric edges, so compare both on ablations.

Concatenate eight 16-dimensional directional summaries to obtain 128 dimensions. Aggregate separate knight and pawn relations to 16 dimensions each, giving a **160-dimensional per-square aggregate**. Concatenate with the existing 64-dimensional square state to obtain 224 dimensions. Project to 64, apply a learned gate, and use a residual update:

`u_i = MLP([h_i, aggregate_i])`  
`h_i_next = LayerNorm(h_i + sigmoid(G([h_i, aggregate_i])) * u_i)`

If king adjacency is not already captured by the ray channels, add a dedicated channel and adjust aggregate dimensions accordingly. For the baseline, adjacent king geometry is available in edge features, so no extra channel is allocated.

**Depth:** three layers initially. One layer captures direct relationships; subsequent layers let information propagate through intermediary squares and pieces. Test one, two, three and four layers rather than assuming greater depth helps.

## 5. Global readout

A simple mean over all 64 squares is insufficient as the sole readout: it loses spatial organization and can dilute important pieces. Use three complementary branches after the last message-passing layer.

| Branch | Operation | Output |
|---|---|---:|
| Piece-aware | For each of 12 colored piece types, masked sum and/or attention pool, then project each group to 32 | 384 |
| Spatial | Divide the board into sixteen fixed 2x2 regions; project square states to 16 and pool per region | 256 |
| King-centered | Two king-node embeddings (2x64) plus learned summaries of each king's surrounding 8 squares (2x32) | 192 |
| **Total** | Concatenation | **832** |

**Empty piece groups:** use a learned zero-group embedding or zeros with an explicit count; attention pooling must handle empty masks without NaNs. Include piece counts so attention pooling does not discard multiplicity. The 2x2 spatial regions are fixed in White-oriented coordinates. For king neighborhoods, use a mask for off-board squares and preserve neighbor direction.

These branches preserve useful *summaries*, not the entire exact position. To guarantee exact position access, concatenate a separately encoded original occupancy tensor (for example a low-rank projection of the flattened 64x13 one-hot board) into the final head, or use an explicit skip path from the raw input. Do not claim that a learned compressed embedding is mathematically lossless. The raw 64x13 one-hot representation itself is lossless for piece placement.

## 6. Material and state pathway

Compute a deterministic White-minus-Black material baseline using pawn=100, knight=320, bishop=330, rook=500, queen=900 centipawns. These are initialization conventions, not fixed truths. Build a 32-dimensional learned feature vector from piece counts (12), castling rights, side to move, phase, and any other selected state features. Project/pad to 32 as needed.

Base global vector: `832 + 32 = 864`. If adding a raw-board skip path, its projected width is additional (e.g. 64), making the head input 928 rather than 864. **Recommended v0.1:** include the 64-dimensional raw-board skip, so the final MLP input is **928**.

Evaluation head: `928 -> 256 -> 64 -> 1`. Interpret its output as a positional correction in pawn units. The scalar White-positive evaluation is:

`V = material_cp / 100 + positional_correction_pawns`.

Do not hard-clip the correction: fortresses, trapped pieces, promotions, mating attacks and endgames can invalidate simplistic material assumptions. Monitor correction magnitude to detect a network that merely relearns material. An optional second head predicts W/D/L from the shared 64-dimensional penultimate representation; it is auxiliary, not a replacement for a search-compatible value convention.

## 7. Tensor shape reference

Assume batch size `B`:

| Stage | Shape |
|---|---|
| Piece IDs | `[B,64]` |
| Square embeddings | `[B,64,64]` |
| Precomputed ray index | `[64,8,7]` |
| Batched ray features/masks | `[B,64,8,7,F_edge]` / `[B,64,8,7]` |
| Directional messages | `[B,64,8,7,16]` |
| Directional aggregates | `[B,64,8,16]` |
| Knight and pawn aggregate | `[B,64,32]` |
| Total aggregate | `[B,64,160]` |
| Updated squares | `[B,64,64]` |
| Piece-aware readout | `[B,384]` |
| Spatial readout | `[B,256]` |
| King readout | `[B,192]` |
| Material/state features | `[B,32]` |
| Raw-board skip | `[B,64]` |
| Head input | `[B,928]` |
| Evaluation | `[B,1]` |

Maximum geometric ray slots per position: `64*8*7=3584`, including invalid off-board slots. Mask invalid entries; for production inference, compress valid edges into static source/destination lists to avoid wasted computation.

## 8. Training specification

Start with supervised evaluation targets from a strong teacher at a documented depth or node budget, and use a separate held-out set split by game or opening family to reduce near-duplicate leakage. Keep target perspective consistent. If using teacher centipawns, transform extreme scores or mate scores carefully and document the transformation. Avoid treating forced-mate centipawns as ordinary finite labels. Include diverse opening, middlegame, endgame, tactical and quiet positions.

Recommended initial losses: robust regression (Huber) on normalized teacher evaluation; optionally cross-entropy for teacher WDL if reliable WDL targets are available. If training a WDL head without WDL targets, use actual game outcomes with appropriate caution about position-level label noise. Record material-only baseline loss and stratify validation by phase, material imbalance, checks, pins, discovered attacks and king exposure.

Use AdamW, gradient clipping, mixed precision where numerically stable, and checkpoint the best held-out model. Hyperparameters and data volumes are experimental choices; benchmark a small subset before committing to a long training run.

## 9. Evaluation and ablations

Compare at least: (A) explicit material + square MLP, (B) three-layer direct-ray GNN, (C) direct + x-ray, (D) C plus structured readout, (E) D plus raw-board skip, and (F) full model with optional WDL. Match training examples and compute as closely as practical. Compare against a reproducible NNUE baseline and, where feasible, a Leela network with clearly documented network size and hardware. Report parameters, inference latency, positions/sec, memory, validation error, and playing strength at fixed search time/nodes. Latency measurements must include graph-feature construction.

**Diagnostic test positions:** pinned pieces; overloaded defenders; x-ray attacks through one or two blockers; rook behind own pawn; open-file rooks; trapped pieces; passed pawns; opposite-side castling; insufficient material; materially imbalanced but tactically winning positions. For each, inspect material baseline and learned correction separately.

**Symmetry tests:** for properly transformed color-swapped positions, the White-positive output should approximately negate. Mirror-file augmentation is not automatically valid when castling rights are present; transform or exclude incompatible cases.

## 10. Inference and engine integration

The primary risk relative to NNUE is that dynamic blocker masks and three rounds of message passing may be expensive per search node. Build a batchable PyTorch reference first, then profile feature construction, message projection, masked aggregation and readout independently. Exploit fixed board geometry, static edge indexing, fused kernels and lower precision only after confirming numerical and chess correctness.

Do not assume NNUE-style incremental updates will be straightforward: moving a blocker can change visibility on several rays, and multiple message-passing layers spread the impact further. A potential later optimization is to recompute only affected rays and their downstream neighborhoods, but this requires correctness tests against full recomputation.

For engine integration, specify whether evaluation is White-positive or side-to-move-positive at the API boundary. Handle terminal mate/stalemate and draw rules in the search engine, not solely through the network.

## 11. Implementation modules and milestones

Suggested code layout:

```
raygnn/
  encoding.py       # Lossless board/state inputs and initial features
  geometry.py       # Fixed ray indices, blocker masks, attacks and x-rays
  layers.py         # Parallel message construction and gated square updates
  readout.py        # Piece, spatial, king and raw-board skip branches
  model.py          # Material pathway, value head, optional WDL head
  train.py          # Data loaders, losses, metrics, checkpointing
  benchmark.py      # Latency, throughput and memory measurements
  tests/            # Geometry, shapes, symmetry, numerical and chess cases
```

1. Implement geometry and compare attack/visibility masks against a trusted chess library over random legal positions.
2. Implement the `[B,64,64]` encoder, one ray layer and a simple head; verify gradients and invariances.
3. Add three layers and the structured readout; confirm all tensor shapes and no NaNs on empty groups.
4. Add explicit material, 64-dimensional raw-board skip and optional WDL head.
5. Train material/MLP/direct-ray/x-ray/readout ablations on identical data.
6. Profile single-position CPU and GPU inference before expensive search integration.
7. Integrate the best variant into a search engine and test at fixed search budgets.

## 12. Open design questions

- Do first-blocker features suffice, or does a clipped blocker count / ordered blocker encoding improve x-ray understanding?
- Should direction aggregation be sum, normalized sum, gated sum, or attention?
- How much information does the raw-board skip add after structured pooling?
- Can two layers achieve similar quality to three at materially lower search cost?
- Does the model learn useful tactical features, or mainly improve teacher-score fitting?
- Can sparse changed-ray recomputation make this competitive with incremental NNUE?

**Decision for v0.1:** implement a three-layer, 64-dimensional ray GNN with 16-dimensional directional messages, a 160-dimensional aggregate, an 832-dimensional structured readout, a 32-dimensional material/state branch, a 64-dimensional raw-board skip and a `928 -> 256 -> 64 -> 1` evaluation head. Treat these widths as starting hyperparameters and the architecture as an experimental specification, not an established improvement.
