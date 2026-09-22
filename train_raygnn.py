"""Train the non-incremental RayGNN reference model on JSONL FEN positions."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from model.modules.raygnn import RayGNNEvaluationNetwork, RayGNNPosition
from model.modules.raygnn_data import (
    RayGNNBinpackBatchProvider,
    RayGNNJsonlDataset,
    collate_raygnn,
)


def parse_duration(value: str) -> float:
    """Parse ``DD:HH:MM:SS``, ``HH:MM:SS``, or a seconds value."""
    if value.isdigit():
        return float(value)
    parts = value.split(":")
    if not 2 <= len(parts) <= 4 or not all(part.isdigit() for part in parts):
        raise argparse.ArgumentTypeError("Use seconds, HH:MM:SS, or DD:HH:MM:SS.")
    values = [int(part) for part in parts]
    if len(values) == 4:
        days, hours, minutes, seconds = values
        return float(days * 86_400 + hours * 3_600 + minutes * 60 + seconds)
    if len(values) == 3:
        hours, minutes, seconds = values
        return float(hours * 3_600 + minutes * 60 + seconds)
    minutes, seconds = values
    return float(minutes * 60 + seconds)


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("datasets", nargs="+", help="JSONL records, or `.binpack` files with --binpack.")
    parser.add_argument("--binpack", action="store_true", help="Read native binpack FEN/state records; repetition count is fixed to zero.")
    parser.add_argument("--binpack-score-scale", type=float, default=100.0, help="Native score units per pawn, default: centipawns.")
    parser.add_argument("--validation-datasets", nargs="*", default=[])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epoch-size", type=int, default=131_072)
    parser.add_argument("--validation-size", type=int, default=262_144)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--check-val-every-n-epoch", type=int, default=5)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--wdl", action="store_true", help="Train the optional WDL head when every record supplies wdl.")
    parser.add_argument("--wdl-weight", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--default-root-dir", type=Path, default=Path("raygnn-runs"))
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--max-time", type=parse_duration, default=None, help="Soft epoch-boundary limit.")
    parser.add_argument("--no-amp", action="store_true")


def move_position(position: RayGNNPosition, device: torch.device) -> RayGNNPosition:
    return position.to(device)


def move_batch(batch, device: torch.device):
    return {
        "position": move_position(batch["position"], device),
        "score_pawns_white": batch["score_pawns_white"].to(device, non_blocking=True),
        "wdl": None if batch["wdl"] is None else batch["wdl"].to(device, non_blocking=True),
    }


def batch_loss(model, batch, use_wdl: bool, wdl_weight: float):
    if use_wdl:
        if batch["wdl"] is None:
            raise ValueError("--wdl requires every training record to include a WDL target.")
        predicted, wdl_logits = model(batch["position"], return_wdl=True)
        wdl_loss = -(batch["wdl"] * F.log_softmax(wdl_logits, dim=1)).sum(dim=1).mean()
    else:
        predicted = model(batch["position"])
        wdl_loss = predicted.new_zeros(())
    value_loss = F.huber_loss(predicted, batch["score_pawns_white"], delta=1.0)
    return value_loss + wdl_weight * wdl_loss, value_loss.detach(), wdl_loss.detach()


@torch.no_grad()
def evaluate(model, loader, batches: int, device: torch.device, use_wdl: bool, wdl_weight: float, amp: bool):
    model.eval()
    total_loss = total_value = total_wdl = 0.0
    iterator = iter(loader)
    for _ in range(batches):
        batch = move_batch(next(iterator), device)
        with torch.autocast(device_type=device.type, enabled=amp):
            loss, value_loss, wdl_loss = batch_loss(model, batch, use_wdl, wdl_weight)
        total_loss += float(loss)
        total_value += float(value_loss)
        total_wdl += float(wdl_loss)
    return {"val_loss": total_loss / batches, "val_value_loss": total_value / batches, "val_wdl_loss": total_wdl / batches}


def save_checkpoint(path: Path, model, optimizer, epoch: int, step: int, args) -> None:
    torch.save({"epoch": epoch, "global_step": step, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "args": vars(args)}, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    args = parser.parse_args()
    if args.epoch_size < args.batch_size or args.epoch_size % args.batch_size:
        parser.error("epoch-size must be a positive multiple of batch-size.")
    if args.validation_datasets and (args.validation_size < args.batch_size or args.validation_size % args.batch_size):
        parser.error("validation-size must be a positive multiple of batch-size.")
    if args.accelerator == "auto":
        args.accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    if args.accelerator == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable.")
    device = torch.device(args.accelerator)
    amp = device.type == "cuda" and not args.no_amp
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    root = args.default_root_dir
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    if args.binpack:
        if args.wdl:
            parser.error("Binpack records expose a game result but not a WDL target; omit --wdl.")
        train_loader = RayGNNBinpackBatchProvider(args.datasets, args.batch_size, args.num_workers, args.binpack_score_scale)
        val_loader = None if not args.validation_datasets else RayGNNBinpackBatchProvider(args.validation_datasets, args.batch_size, args.num_workers, args.binpack_score_scale)
    else:
        train_loader = DataLoader(RayGNNJsonlDataset(args.datasets), batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=amp, collate_fn=collate_raygnn)
        val_loader = None if not args.validation_datasets else DataLoader(RayGNNJsonlDataset(args.validation_datasets), batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=amp, collate_fn=collate_raygnn)
    model = RayGNNEvaluationNetwork(layers=args.layers, with_wdl=args.wdl).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    train_batches = args.epoch_size // args.batch_size
    val_batches = args.validation_size // args.batch_size
    started = time.monotonic()
    global_step = 0
    metrics_path = root / "metrics.csv"
    try:
        with metrics_path.open("w", newline="") as metrics_file:
            writer = csv.DictWriter(
                metrics_file,
                fieldnames=(
                    "epoch", "step", "train_loss", "train_value_loss",
                    "train_wdl_loss", "val_loss", "val_value_loss", "val_wdl_loss",
                ),
            )
            writer.writeheader()
            for epoch in range(args.max_epochs):
                model.train()
                running = torch.zeros(3, device=device)
                train_iterator = iter(train_loader)
                for _ in range(train_batches):
                    batch = move_batch(next(train_iterator), device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=amp):
                        loss, value_loss, wdl_loss = batch_loss(model, batch, args.wdl, args.wdl_weight)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    running += torch.stack((loss.detach(), value_loss, wdl_loss))
                    global_step += 1
                running = (running / train_batches).tolist()
                row = {
                    "epoch": epoch + 1, "step": global_step, "train_loss": running[0],
                    "train_value_loss": running[1], "train_wdl_loss": running[2],
                    "val_loss": "", "val_value_loss": "", "val_wdl_loss": "",
                }
                if val_loader is not None and (epoch + 1) % args.check_val_every_n_epoch == 0:
                    row.update(evaluate(model, val_loader, val_batches, device, args.wdl, args.wdl_weight, amp))
                writer.writerow(row)
                metrics_file.flush()
                print(json.dumps(row), flush=True)
                if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.max_epochs:
                    save_checkpoint(root / f"epoch={epoch + 1}-step={global_step}.ckpt", model, optimizer, epoch + 1, global_step, args)
                if args.max_time is not None and time.monotonic() - started >= args.max_time:
                    break
    finally:
        for loader in (train_loader, val_loader):
            if loader is not None and hasattr(loader, "close"):
                loader.close()


if __name__ == "__main__":
    main()
