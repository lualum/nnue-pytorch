#!/usr/bin/env python3
"""Audit OrderedRay2 gating, quantization survival, and prediction impact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import data_loader
import model as M


FEATURES = "Full_Threats+PP_3Wide+OrderedRay2+HalfKAv2_hm^"


def summary(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().cpu().flatten()
    abs_values = values.abs()
    return {
        "min": values.min().item(),
        "max": values.max().item(),
        "mean": values.mean().item(),
        "std": values.std().item(),
        "abs_mean": abs_values.mean().item(),
        "abs_p50": abs_values.quantile(0.50).item(),
        "abs_p90": abs_values.quantile(0.90).item(),
        "abs_p99": abs_values.quantile(0.99).item(),
    }


def load_checkpoint(path: Path):
    config = M.NNUEConfig(features=FEATURES)
    network = M.NNUE(config=config)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    network.load_state_dict(checkpoint["state_dict"])
    network.eval()
    return network


@torch.no_grad()
def prediction_differences(network, validation_data: Path, batches: int, batch_size: int):
    provider = data_loader.SparseBatchProvider(
        FEATURES, [str(validation_data)], batch_size, cyclic=True, num_workers=1
    )
    ray = network.model.input.features[2]
    original_gate = ray.gate.detach().clone()
    full_precision, fake_quantized = [], []
    for _ in range(batches):
        batch = next(provider)
        us, them, white, black, _, _, piece_count = batch
        ray.gate.copy_(original_gate)
        enabled_fp = network.model(us, them, white, black, piece_count, False, False)
        enabled_fq = network.model(us, them, white, black, piece_count, True, True)
        ray.gate.zero_()
        disabled_fp = network.model(us, them, white, black, piece_count, False, False)
        disabled_fq = network.model(us, them, white, black, piece_count, True, True)
        full_precision.append((enabled_fp - disabled_fp).flatten().cpu())
        fake_quantized.append((enabled_fq - disabled_fq).flatten().cpu())
    ray.gate.copy_(original_gate)

    def diff_summary(parts):
        values = torch.cat(parts).float() * network.model.quantization.nnue2score
        return {
            **summary(values),
            "rmse": values.square().mean().sqrt().item(),
            "max_abs": values.abs().max().item(),
            "fraction_nonzero": values.ne(0).float().mean().item(),
            "positions": values.numel(),
            "units": "Stockfish evaluation units",
        }

    return {
        "full_precision": diff_summary(full_precision),
        "fake_quantized": diff_summary(fake_quantized),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    network = load_checkpoint(args.checkpoint)
    reference = torch.load(args.reference_model, map_location="cpu", weights_only=False)
    ray = network.model.input.features[2]
    initial_ray = reference.model.input.features[2]
    effective = ray.get_export_weights()
    quantized = network.model.quantization.quantize_feature_transformer_weights(
        effective, torch.int8
    )
    raw_delta = ray.weight.detach() - initial_ray.weight.detach()

    result = {
        "gate": ray.gate.item(),
        "initial_gate": initial_ray.gate.item(),
        "raw_weight": summary(ray.weight),
        "effective_weight_gate_times_weight": summary(effective),
        "exported_int8": {
            **summary(quantized),
            "nonzero": torch.count_nonzero(quantized).item(),
            "total": quantized.numel(),
            "fraction_nonzero": quantized.ne(0).float().mean().item(),
            "fraction_saturated": quantized.abs().eq(127).float().mean().item(),
        },
        "learning_from_pre_warmup": {
            "raw_weight_delta": summary(raw_delta),
            "l2_norm": raw_delta.float().norm().item(),
            "fraction_changed": raw_delta.ne(0).float().mean().item(),
            "gate_opened": ray.gate.item() != initial_ray.gate.item(),
        },
        "prediction_difference_rays_enabled_minus_disabled": prediction_differences(
            network, args.validation_data, args.batches, args.batch_size
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
