"""Portable weight container for the movement evaluator runtime."""

import struct

import numpy as np

from ..model import NNUEModel


MOVEMENT_MAGIC = b"SFMMNN01"
MOVEMENT_VERSION = 2


class MovementNNUEWriter:
    """Serialize named float32 tensors for a small native movement runtime.

    Geometry is deterministic and is therefore not stored. Every tensor record
    contains its state-dict name and shape so a native loader can reject a model
    with a mismatched layout instead of silently reading incorrect weights.
    """

    def __init__(self, model: NNUEModel, description: str | None = None):
        if model.network_type != "movement" or model.movement is None:
            raise ValueError(".mnnue export requires a movement network.")

        if description is None:
            description = "Lightweight movement NNUE"

        movement = model.movement
        if movement.dim != 8 or movement.iterations != 3:
            raise ValueError(
                "The current Stockfish movement runtime requires dimension 8 "
                "and 3 iterations."
            )
        tensors = movement.state_dict()
        encoded_description = description.encode("utf-8")

        self.buf = bytearray(MOVEMENT_MAGIC)
        self.buf.extend(struct.pack("<I", MOVEMENT_VERSION))
        self.buf.extend(struct.pack("<I", movement.dim))
        self.buf.extend(struct.pack("<I", movement.iterations))
        self.buf.extend(struct.pack("<f", model.quantization.nnue2score))
        self.buf.extend(struct.pack("<I", len(encoded_description)))
        self.buf.extend(encoded_description)
        self.buf.extend(struct.pack("<I", len(tensors)))

        for name, tensor in tensors.items():
            encoded_name = name.encode("utf-8")
            values = tensor.detach().cpu().numpy().astype("<f4", copy=False)
            self.buf.extend(struct.pack("<I", len(encoded_name)))
            self.buf.extend(encoded_name)
            self.buf.extend(struct.pack("<I", values.ndim))
            for dimension in values.shape:
                self.buf.extend(struct.pack("<I", dimension))
            self.buf.extend(values.tobytes(order="C"))


__all__ = ["MOVEMENT_MAGIC", "MOVEMENT_VERSION", "MovementNNUEWriter"]
