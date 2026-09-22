"""Streaming, state-preserving training data for the RayGNN reference model.

One JSON object per line is required::

    {"fen": "... six-field FEN ...", "repetition_count": 0,
     "score_cp_white": 34}

``score_cp_white`` is a Stockfish-like centipawn score from White's
perspective.  It is deliberately not side-to-move normalized, matching the
RayGNN value contract.  ``wdl`` is optional and, when supplied, is a three
element White-perspective probability vector ``[win, draw, loss]``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import IterableDataset, get_worker_info

from .raygnn import RayGNNPosition


class RayGNNJsonlDataset(IterableDataset):
    """Repeat a collection of JSONL files without loading it into memory."""

    def __init__(self, filenames: Sequence[str | Path]):
        super().__init__()
        self.filenames = tuple(Path(filename) for filename in filenames)
        if not self.filenames:
            raise ValueError("At least one RayGNN JSONL data file is required.")
        for filename in self.filenames:
            if not filename.is_file():
                raise FileNotFoundError(filename)

    @staticmethod
    def _validate(record: dict[str, Any], filename: Path, line_number: int) -> dict[str, Any]:
        if not isinstance(record, dict):
            raise ValueError(f"{filename}:{line_number} must contain a JSON object.")
        required = {"fen", "score_cp_white"}
        missing = required - record.keys()
        if missing:
            raise ValueError(f"{filename}:{line_number} is missing {sorted(missing)}.")
        if not isinstance(record["fen"], str):
            raise ValueError(f"{filename}:{line_number} has a non-string FEN.")
        try:
            record["score_cp_white"] = float(record["score_cp_white"])
            record["repetition_count"] = int(record.get("repetition_count", 0))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{filename}:{line_number} has invalid numeric fields.") from error
        if record["repetition_count"] < 0:
            raise ValueError(f"{filename}:{line_number} has a negative repetition_count.")
        if "wdl" in record:
            wdl = record["wdl"]
            if not isinstance(wdl, list) or len(wdl) != 3:
                raise ValueError(f"{filename}:{line_number} wdl must have three values.")
            record["wdl"] = [float(value) for value in wdl]
        return record

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        while True:
            emitted = 0
            global_line = 0
            for filename in self.filenames:
                with filename.open(encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        if global_line % worker_count != worker_id:
                            global_line += 1
                            continue
                        global_line += 1
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as error:
                            raise ValueError(f"Invalid JSON at {filename}:{line_number}.") from error
                        yield self._validate(record, filename, line_number)
                        emitted += 1
            if emitted == 0:
                raise RuntimeError("No JSONL positions were assigned to this data-loader worker.")


def collate_raygnn(records: list[dict[str, Any]]) -> dict[str, torch.Tensor | RayGNNPosition | None]:
    """Turn JSONL records into a RayGNNPosition and White-positive targets."""
    if not records:
        raise ValueError("Cannot collate an empty RayGNN batch.")
    position = RayGNNPosition.from_fens(
        [record["fen"] for record in records],
        repetition_count=[record["repetition_count"] for record in records],
    )
    result: dict[str, torch.Tensor | RayGNNPosition | None] = {
        "position": position,
        "score_pawns_white": torch.tensor(
            [record["score_cp_white"] / 100.0 for record in records], dtype=torch.float32
        ).unsqueeze(1),
        "wdl": None,
    }
    has_wdl = ["wdl" in record for record in records]
    if any(has_wdl) and not all(has_wdl):
        raise ValueError("A batch cannot mix records with and without WDL targets.")
    if all(has_wdl):
        wdl = torch.tensor([record["wdl"] for record in records], dtype=torch.float32)
        if torch.any(wdl < 0) or not torch.allclose(wdl.sum(dim=1), torch.ones(len(records))):
            raise ValueError("Every WDL target must be non-negative and sum to one.")
        result["wdl"] = wdl
    return result


def collate_binpack_raygnn(records: list[tuple[str, int, int]], score_scale: float = 100.0):
    """Convert binpack's side-to-move scores to RayGNN's White-positive value.

    Binpack preserves FEN fields needed for board, side to move, castling,
    en-passant, and halfmove clock. It does not preserve repetition history, so
    this feasibility path explicitly supplies zero repetition count.
    """
    if score_scale <= 0:
        raise ValueError("score_scale must be positive.")
    position = RayGNNPosition.from_fens([record[0] for record in records])
    scores_stm = torch.tensor([record[1] for record in records], dtype=torch.float32)
    white_sign = position.side_to_move.mul(2).sub(1)
    return {
        "position": position,
        "score_pawns_white": (scores_stm * white_sign / score_scale).unsqueeze(1),
        "wdl": None,
    }


class RayGNNBinpackBatchProvider:
    """Native `.binpack` stream retaining labels alongside state-complete FENs."""

    def __init__(self, filenames, batch_size: int, concurrency: int = 1, score_scale: float = 100.0):
        from data_loader import stream
        from data_loader.config import DataloaderDDPConfig, DataloaderSkipConfig

        self._stream_api = stream
        self._score_scale = score_scale
        self._stream = stream.create_fen_batch_stream(
            concurrency, list(filenames), batch_size, True,
            DataloaderSkipConfig(), DataloaderDDPConfig(rank=0, world_size=1),
        )

    def __iter__(self):
        return self

    def __next__(self):
        batch = self._stream_api.fetch_next_fen_batch(self._stream)
        if not batch:
            raise StopIteration
        try:
            return collate_binpack_raygnn(batch.contents.get_records(), self._score_scale)
        finally:
            self._stream_api.destroy_fen_batch(batch)

    def close(self):
        if self._stream is not None:
            self._stream_api.destroy_fen_batch_stream(self._stream)
            self._stream = None
