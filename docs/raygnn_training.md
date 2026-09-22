# RayGNN training

RayGNN is trained separately from `train.py` because the legacy `.binpack`
loader currently discards its state fields. The packed records themselves do
retain board, side-to-move, castling rights, en-passant target, halfmove clock,
teacher score, and game result. They do not retain repetition history.

## Dataset format

Use UTF-8 JSONL. Every line must contain a legal six-field FEN and a teacher
evaluation in **White-positive centipawns**. Repetition is supplied separately
because it is not part of FEN.

Mate labels must be transformed before writing `score_cp_white`; do not write
Stockfish's finite mate sentinel as an ordinary centipawn target.

```json
{"fen":"r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1","repetition_count":0,"score_cp_white":34}
```

For auxiliary WDL training, add a White-perspective probability vector:

```json
{"fen":"...","repetition_count":1,"score_cp_white":-80,"wdl":[0.18,0.41,0.41]}
```

All records in a WDL run must have `wdl`; it must be `[White win, draw, Black
win]`, non-negative and sum to one.

## Kaggle reference command

This preserves the reference notebook's T4-oriented batch size, epoch size,
80-epoch budget, seed, single worker and epoch-boundary time limit:

```bash
TORCH_COMPILE_DISABLE=1 python train_raygnn.py /kaggle/input/raygnn-data/train.binpack \
  --binpack --validation-datasets /kaggle/input/raygnn-data/valid.binpack \
  --batch-size 256 --epoch-size 131072 --validation-size 262144 \
  --max-epochs 80 --check-val-every-n-epoch 5 --accelerator cuda \
  --num-workers 1 --seed 42 --max-time 00:01:15:00 \
  --default-root-dir /kaggle/working/raygnn-runs/run-1
```

The `.binpack` path converts side-to-move teacher scores to White-positive
pawns and explicitly uses `repetition_count=0`. It is suitable for the first
geometry/material feasibility run, but not for measuring whether the model
uses repetition information. Use the JSONL path for that experiment.

Use a validation file split by game or opening family. A separate random stream
from the same file only detects training instability and is not a generalization
measurement.
