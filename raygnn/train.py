"""Supervised reference trainer for JSONL FEN/teacher-score records."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import chess
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .encoding import boards_to_batch
from .model import RayGNN, RayGNNConfig


class TeacherDataset(Dataset):
    def __init__(self, path: Path, validation: bool, validation_fraction: float = 0.1):
        self.rows = []
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "mate" in row or row.get("eval_cp") is None:
                continue
            group = row.get("game_id", row.get("opening_family"))
            if group is None:
                raise ValueError(f"line {line_number}: game_id or opening_family is required for grouped holdout")
            bucket = int.from_bytes(hashlib.sha256(str(group).encode()).digest()[:8], "big") / 2**64
            if (bucket < validation_fraction) != validation:
                continue
            self.rows.append(row)
        if not self.rows:
            raise ValueError(f"no {'validation' if validation else 'training'} examples in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        board = chess.Board(row["fen"])
        # Teacher score is White-positive centipawns. Smoothly compress extreme scores.
        target = 10 * math.tanh(float(row["eval_cp"]) / 1000)
        wdl = row.get("wdl")
        if wdl is not None and (len(wdl) != 3 or min(wdl) < 0 or sum(wdl) <= 0):
            raise ValueError("wdl must contain nonnegative White/Draw/Black weights")
        return board, target, wdl


def collate(rows):
    boards, targets, wdls = zip(*rows)
    wdl = None
    if all(value is not None for value in wdls):
        wdl = torch.tensor(wdls, dtype=torch.float32)
        wdl = wdl / wdl.sum(-1, keepdim=True)
    return boards_to_batch(list(boards)), torch.tensor(targets, dtype=torch.float32)[:, None], wdl


def evaluate(model: RayGNN, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    baseline_sum = 0.0
    count = 0
    with torch.inference_mode():
        for batch, target, _ in loader:
            output = model(batch.to(device))
            target = target.to(device)
            loss_sum += nn.functional.huber_loss(output.value, target, reduction="sum").item()
            baseline_sum += nn.functional.huber_loss(output.material, target, reduction="sum").item()
            count += target.numel()
    return loss_sum / count, baseline_sum / count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="JSONL records with fen, White-positive eval_cp and game_id/opening_family")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--direct-only", action="store_true")
    parser.add_argument("--first-piece-only", action="store_true")
    parser.add_argument("--simple-readout", action="store_true")
    parser.add_argument("--no-raw-skip", action="store_true")
    parser.add_argument("--wdl-head", action="store_true")
    args = parser.parse_args()
    config = RayGNNConfig(layers=args.layers, use_xray=not args.direct_only,
                          first_piece_only=args.first_piece_only,
                          structured_readout=not args.simple_readout,
                          raw_board_skip=not args.no_raw_skip, wdl_head=args.wdl_head)
    train_set = TeacherDataset(args.data, False)
    val_set = TeacherDataset(args.data, True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=collate)
    model = RayGNN(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))
    best = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        train_sum = 0.0
        for batch, target, wdl in train_loader:
            batch, target = batch.to(args.device), target.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=args.device.startswith("cuda")):
                output = model(batch)
                loss = nn.functional.huber_loss(output.value, target)
                if config.wdl_head:
                    if wdl is None:
                        raise ValueError("--wdl-head requires wdl targets on every record")
                    loss = loss - (wdl.to(args.device) * nn.functional.log_softmax(output.wdl_logits, -1)).sum(-1).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_sum += loss.item() * target.shape[0]
        validation, material = evaluate(model, val_loader, args.device)
        print(f"epoch={epoch + 1} train={train_sum / len(train_set):.5f} validation={validation:.5f} material={material:.5f}")
        if validation < best:
            best = validation
            torch.save({"model": model.state_dict(), "config": config.__dict__, "validation_huber": best,
                        "target": "10*tanh(White-positive teacher centipawns/1000)"}, args.output)


if __name__ == "__main__":
    main()
