#!/usr/bin/env python3
"""Create matched baseline/candidate .pt models from an official Stockfish net."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

# Direct script execution places scripts/ rather than the repository root on
# sys.path.  Add the checkout root so this works from Kaggle and locally.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import model as M


BASELINE_FEATURES = "Full_Threats+PP_3Wide+HalfKAv2_hm^"
CANDIDATE_FEATURES = "Full_Threats+PP_3Wide+OrderedRay2+HalfKAv2_hm^"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Official baseline .nnue")
    parser.add_argument("baseline", type=Path, help="Output baseline .pt")
    parser.add_argument("candidate", type=Path, help="Output OrderedRay2 .pt")
    args = parser.parse_args()

    baseline_config = M.NNUEConfig(features=BASELINE_FEATURES)
    print("Reading official Stockfish network...", flush=True)
    with args.source.open("rb") as source:
        reader = M.NNUEReader(
            source,
            BASELINE_FEATURES,
            config=baseline_config.model_config,
        )
    baseline = M.NNUE(config=baseline_config)
    baseline.model = reader.model

    print("Constructing zero-ray candidate...", flush=True)
    torch.manual_seed(20261002)
    candidate_config = M.NNUEConfig(features=CANDIDATE_FEATURES)
    candidate = M.NNUE(config=candidate_config)

    # Both models begin with exactly the same Stockfish parameters. The new
    # feature is inserted before HalfKAv2 in the serialized int8/int16 layout
    # and starts at zero, making the initial evaluations identical.
    candidate.model.input.bias.data.copy_(baseline.model.input.bias.data)
    candidate.model.input.features[0].weight.data.copy_(
        baseline.model.input.features[0].weight.data
    )
    candidate.model.input.features[1].weight.data.copy_(
        baseline.model.input.features[1].weight.data
    )
    candidate.model.input.features[2].weight.data.zero_()
    candidate.model.input.features[3].load_export_weights(
        baseline.model.input.features[2].get_export_weights()
    )
    print("Copying matched parameters...", flush=True)
    candidate.model.layer_stacks.load_state_dict(
        baseline.model.layer_stacks.state_dict()
    )

    # Fail closed: before training, the expanded model must be exactly the
    # parent model plus a zero-contribution OrderedRay2 block.
    assert torch.count_nonzero(candidate.model.input.features[2].weight) == 0
    assert torch.equal(candidate.model.input.bias, baseline.model.input.bias)
    assert torch.equal(
        candidate.model.input.features[0].weight,
        baseline.model.input.features[0].weight,
    )
    assert torch.equal(
        candidate.model.input.features[1].weight,
        baseline.model.input.features[1].weight,
    )
    assert torch.equal(
        candidate.model.input.features[3].get_export_weights(),
        baseline.model.input.features[2].get_export_weights(),
    )
    print("Verified exact parent copy with zero-gated OrderedRay2 weights.", flush=True)

    args.baseline.parent.mkdir(parents=True, exist_ok=True)
    args.candidate.parent.mkdir(parents=True, exist_ok=True)
    torch.save(baseline, args.baseline)
    torch.save(candidate, args.candidate)

    print(f"baseline={args.baseline}")
    print(f"candidate={args.candidate}")


if __name__ == "__main__":
    main()
