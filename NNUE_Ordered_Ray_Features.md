# NNUE ordered ray features: design and FullThreats decision

Date: October 1, 2026

Status: proposed experiment, not a demonstrated playing-strength improvement. Stockfish references below describe the SF19 development cycle reviewed for this design.

## 1. Decision

**Retain FullThreats initially. Add a sparse feature family for ordered slider-blocker-target triples. Do not replace FullThreats with these triples.**

The two representations cover different information. Direct threats describe immediate piece relationships, including knight and pawn attacks. Ordered ray triples describe a slider's relationship with a second piece through exactly one intervening piece. They cannot replace all direct threats, and a feature family that requires two occupied squares misses even ordinary slider attacks when no second piece exists.

Keep the existing piece-square and pawn-pair inputs as well. Train a single evaluator jointly, with one output and the existing dense head. This is an input representation extension, not a separately evaluated engine or a score correction added after evaluating Stockfish.

Only investigate partial replacement after the additive model demonstrates value. Full replacement would require a broader representation that explicitly preserves useful direct relationships and independently passes strength testing.

## 2. What SF19 establishes

The most relevant precedent is SFNNv16's PP_3Wide change [1]. It added pawn-pair inputs for pawns on the same or adjacent files. The developers narrowed the representation after observing which pairs mattered in an all-pawn-pairs model. Because the new representation covered pawn-pawn relationships already present in threat inputs, the overlapping inputs were removed. HalfKA remained unchanged.

The architecture and new network together failed STC but passed LTC, VLTC, and VVLTC. The commit reports an initial slowdown around 10%, reduced to about 3.5% on a contributor's machine after optimization. These are results for that submitted patch, not a prediction for ray features.

Other relevant developments include removal of piece-to-king and king-to-piece threat inputs, addition of opposed-pawn inputs, and changes to the small dense head [2, 3]. SF19's release notes also describe quantization-aware training and consistently rescored training data [4]. The release's total Elo improvement must not be attributed to any single feature family.

The experimental lesson is to select useful relationships, preserve fast incremental updates, remove proven overlap, and judge the compiled evaluator in search.

## 3. Proposed feature: OrderedRay2

For each bishop, rook, or queen and each movement-compatible direction:

1. Locate the first occupied square on the ray, B.
2. Continue beyond B and locate the next occupied square, C.
3. If both exist, emit a feature for the ordered triple (S, B, C), where S is the source slider.

The spaces between S and B and between B and C must be empty. Colors do not terminate extraction. A friendly blocker is as relevant as an enemy blocker.

| Triple | Geometric information |
| --- | --- |
| Friendly bishop, enemy knight, enemy king | Absolute pin |
| Friendly rook, enemy bishop, enemy queen | Potential relative pin |
| Friendly rook, enemy king, enemy queen | Skewer geometry |
| Friendly bishop, friendly knight, enemy queen | Potential discovered attack |
| Friendly rook, friendly bishop, enemy king | Potential discovered check |
| Friendly rook, friendly rook, enemy piece | Battery or blocked pressure |

Do not assign a fixed tactical bonus or claim the blocker can legally move. Geometry is input; value is learned. A pinned slider, a defended target, or a blocker unable to leave the ray can change the practical significance.

**Search caveat:** Stockfish normally does not call static evaluation while in check. A slider directly attacking an enemy king may therefore occur only in positions outside the normal evaluation distribution, depending on whose turn it is. In particular, the king-as-first-blocker skewer category should be measured for actual training and inference frequency, not assumed to be a major source of gain. Pins and latent discoveries can occur in ordinary evaluated positions.

## 4. Initial encoding

Use one joint categorical key rather than separate pair features that lose the identity of the shared blocker:

```text
(source_relative_color,
 source_type_and_direction,
 blocker_relative_piece_code,
 target_relative_piece_code,
 source_to_blocker_distance_bucket,
 blocker_to_target_distance_bucket)
```

- Relative color means friendly or enemy with respect to the accumulator perspective.
- Source type and direction has 16 valid combinations: four bishop directions, four rook directions, eight queen directions.
- Blocker and target each have 12 codes: six piece types times two relative colors.
- Distances are square-to-square step counts along the ray, not empty-square counts.
- Initial distance buckets: 1, 2–3, 4–7 steps.
- No NONE codes are necessary because both pieces must exist.
- Transform directions consistently with the baseline's board orientation and perspective convention. Reuse tested orientation functions rather than duplicating them.

The uncompressed feature space is:

```text
2 × 16 × 12 × 12 × 3 × 3 = 41,472 keys
```

Some combinations cannot occur. Start with collision-free dense indexing; compress impossible combinations later if worthwhile. Preserve feature multiplicity: two rays sharing a key contribute twice. Do not convert active IDs into a set.

This compact encoding is translation invariant and loses exact board location. That is an intentional first experiment, not complete geometry. Baseline piece-square features retain location, but their presence does not guarantee that the head can bind every motif to its correct location. If the compact version helps, compare a location-aware version using an exact source square or a small board-region index. Report its increased table size and update cost separately.

Avoid adding several overlapping motif tables immediately. First test one joint table; distance-free keys or training-only factorization can be separate ablations if data sparsity becomes a problem.

## 5. Integration into NNUE

Conceptually, for each perspective c:

```text
A_c = bias + P_c + T_c + PP_c + R_c
output = existing_head(existing_transform(A_stm, A_opponent))
```

P denotes piece-square features, T existing threat features, PP pawn pairs, and R ordered rays. This is a conceptual grouping: preserve the actual baseline's accumulator layout, scaling, activation transform, and output branches.

### Primary experiment: shared feature-transformer width

Give each ray key a learned vector with the baseline accumulator width. Maintain ray contributions separately if their update/refresh lifecycle differs, then combine before the existing activation transform. Keep the dense head, search, and output calibration unchanged.

Use the existing relational-feature quantization conventions where compatible. Try int8 ray weights with int16 or wider accumulation, but prove accumulator bounds and validate exported inference. Never assume int16 is safe solely because Stockfish uses it elsewhere.

At 1,024 dimensions, the 41,472-key ray table alone requires approximately 40.5 MiB with int8 weights, or 81 MiB with int16 weights. This excludes optimizer state and other parameters. Shared-width integration is a clean experiment, not automatically the cheapest one.

### Cost alternative: narrow ray branch

If full-width updates are too expensive, compare a 64-dimensional ray accumulator per perspective, concatenated with the baseline transformed representation before a jointly trained head. Its table is about 2.53 MiB at int8 or 5.06 MiB at int16.

This changes the head and its capacity. Include a baseline with comparable added head capacity when interpreting gains. Neither configuration has a demonstrated advantage before testing.

## 6. Incremental extraction and updates

First implement full extraction as the correctness oracle. Then implement dirty-ray updates.

For each move:

1. Collect every square whose occupancy or piece identity changes: source, destination, en-passant capture square, and both rook squares for castling where applicable.
2. In both the old and new positions, select slider rays aligned with any changed square.
3. Include every valid ray of a moved, captured, or promoted slider, even if alignment selection would miss it.
4. Recompute selected ray descriptors and compare old/new feature-ID multisets.
5. Subtract removed vectors and add new vectors, preserving multiplicities.

Conservative selection may include unchanged rays; that is correct. Do not update only the moving piece, since moving a blocker changes other pieces' rays. Track piece identity changes on occupied destination squares, not just occupancy-bit changes.

Reuse the baseline's lazy evaluation and search-stack conventions. Restore prior accumulator state on undo; do not introduce repeated floating-point add/subtract drift. Handle horizontal-orientation changes explicitly if a king crossing the central files changes feature indexing. A null move changes side-to-move ordering, but not board-derived feature IDs.

Measure full refresh cost, dirty-ray selection cost, average and high-percentile feature changes per move, table-cache behavior, and whole-search throughput. Extraction may dominate even when vector arithmetic is cheap.

## 7. FullThreats: retention and replacement rules

| Component | Initial decision | Replacement condition |
| --- | --- | --- |
| Knight direct relationships | Retain | A separate representation must cover them and pass tests; OrderedRay2 cannot. |
| Pawn-to-piece relationships retained by SF19 | Retain | OrderedRay2 does not generally cover them. |
| Slider direct relationships | Retain | A two-blocker-only representation misses attacks with no second piece and is not an exact substitute. |
| Existing pawn-pair inputs | Retain | Pawn structure is outside OrderedRay2's scope. |
| King-related latent alignments | Add experimentally | Distinct from direct king threats removed during SF19 development. |
| Duplicate ray descriptions | Initially retain directed descriptions | Canonicalize only if the removed description is recoverable and strength/speed tests justify it. |

There is a second obstacle to replacement: the compact triple key omits exact source and target squares, whereas direct threat inputs can preserve them. Even when a triple includes a directly attacked blocker, it is not necessarily an information-preserving substitute for that threat feature.

Do not silently suppress a direct threat whenever a triple happens to exist. That changes its encoding depending on a third piece and may increase feature churn. It requires its own retrained ablation.

If additive OrderedRay2 succeeds, an optional later experiment can define a unified slider relationship representation with explicit direct-edge features for all first blockers and separate x-ray context for second blockers. Compare this against FullThreats plus OrderedRay2. Retain knight and pawn relations. This is a partial reorganization, not evidence that all of FullThreats should disappear.

## 8. Space features are a separate experiment

After the triple experiment, optionally add:

```text
(source_relative_piece, source_square, direction, visible_empty_square_count)
```

Visible empty squares stop at the first blocker or board edge. This exposes open reach; it is not safe mobility, legal mobility, or a complete measure of positional space. Keep this ablation separate so it does not obscure whether blocker triples work. Defer attack-map-dependent safe mobility because its update dependencies are broader.

## 9. Training and experiment sequence

Pin exact engine and trainer commits, dataset manifests, filtering, label convention, quantization scales, and seeds. Use a versioned baseline with piece-square, threat, and pawn-pair features rather than comparing only against a plain historical NNUE.

| Stage | Experiment | Question |
| --- | --- | --- |
| A | Baseline reproduction | Can the pipeline reproduce a stable, functioning evaluator? |
| B | Baseline + triples with an enemy king at B or C | Do restricted king alignments help? Measure category frequency. |
| C | Baseline + triples with enemy non-king C | Do relative pins and discoveries add value? |
| D | Baseline + all two-blocker triples | Does broader coverage justify added cost? |
| E | Best triple model with alternative width or location encoding | What representation gives the best strength-cost tradeoff? |
| F | Best model + ray-reach space features | Is space information independently useful? |
| G | Selective threat-removal ablation | Is any proposed replacement actually beneficial? |

Keep training exposure, labels, data order where practical, and evaluation sets matched. Split by game or source group, not only random positions from the same games. Record duplicate and near-duplicate risks. Report global validation metrics and separate pin/discovery-rich subsets; tactical subsets are diagnostic, not the final objective.

For an inexpensive screening run, initialize compatible baseline weights identically and initialize the new shared-width ray table to zero. At step zero, predictions should match the baseline. Fine-tune all weights, and also fine-tune the baseline control for the same budget. Promising results should be checked across seeds. A weak fine-tuning result alone does not prove the representation cannot work when trained jointly from scratch.

Do not compare raw losses from different target scales or loss definitions. Include quantization-aware training or a validated equivalent deployment-aware procedure. Verify float-to-integer error and score calibration before games.

## 10. Correctness and acceptance criteria

Required correctness checks:

- Full extraction and incremental updates agree over random legal games.
- Captures, promotions, castling, en passant, null moves, undo, and orientation changes are covered.
- Zero ray weights reproduce the unmodified evaluator.
- Repeated feature IDs accumulate with correct counts.
- Integer accumulation cannot overflow under supported legal positions.
- Engine and trainer emit identical feature IDs for the same positions and perspectives.

Required performance evidence:

- Exported CPU inference and complete-search throughput on the intended hardware.
- Held-out error using identical targets and loss, plus subgroup diagnostics.
- Paired-opening games with colors reversed and otherwise identical engine settings.
- Equal-time tests as the deployment criterion, with fixed-node tests only to diagnose evaluation quality independently of speed.
- Short and longer time controls, because the SFNNv16 patch demonstrates that outcomes can differ with time control.
- Confidence intervals or a predeclared sequential test; reserve fresh openings for confirmation after selecting the best candidate.

Do not promise a particular Elo gain or accept a model because a small match happens to win. If accuracy improves but equal-time strength falls, optimize or narrow the feature family. If OrderedRay2 never improves the established baseline under a reasonable matched budget, retain FullThreats and stop expanding this design.

## 11. Initial implementation scope

Implement OrderedRay2 extraction, a collision-free feature index, trainer integration, a shared-width accumulator contribution, incremental updates, quantized export, and the baseline/B/C/D ablations. Keep search and the dense head fixed. Add the narrow branch, location features, space features, and replacement experiments only when a concrete result motivates them.

**Default deliverable: existing SF19-style inputs plus OrderedRay2, trained as one evaluator. FullThreats remains enabled.**

## Sources

1. [SFNNv16 pawn-pair commit, f4bcd404](https://github.com/official-stockfish/Stockfish/commit/f4bcd40409f94bd397a083c5d6243bac6dcc6d85). Feature selection, replacement of overlapping pawn inputs, reported slowdown, and time-control test results.
2. [Official NNUE documentation and architecture history](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html). Sparse accumulation, threat inputs, quantization, and architecture timeline. This is a living document; pin source revisions during implementation.
3. [SFNNv15 commit, e33bb26e](https://github.com/official-stockfish/Stockfish/commit/e33bb26ee7320629dd9114a936a38e6f4d43d52a). Dense-head changes and combined training/data update.
4. [Stockfish 19 release announcement](https://stockfishchess.org/blog/2026/stockfish-19/). Released September 5, 2026; SFNNv16, QAT, training changes, and aggregate release results.

All OrderedRay2 design choices and acceptance recommendations above are proposals derived from these precedents. The cited sources do not establish that OrderedRay2 improves Stockfish.
