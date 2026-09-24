"""Seeded, fixed-size supervised RayGNN experiment runner for JSONL FEN records."""

import argparse
import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import chess
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler

from .encoding import boards_to_batch, swap_colors_reflect_ranks
from .model import RayGNN, RayGNNConfig


def _bucket(group: str) -> float:
    return int.from_bytes(hashlib.sha256(group.encode()).digest()[:8], "big") / 2**64


def load_splits(path: Path, validation_fraction: float = 0.1):
    """Group by game, then remove positions present across splits."""
    train, validation = [], []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "mate" in row or row.get("eval_cp") is None:
            continue
        group = row.get("game_id", row.get("opening_family"))
        if group is None:
            raise ValueError(f"line {line_number}: game_id or opening_family is required")
        board = chess.Board(row["fen"])
        key = " ".join(board.fen(en_passant="fen").split()[:4])
        item = (row, key)
        (validation if _bucket(str(group)) < validation_fraction else train).append(item)
    train_keys = {key for _, key in train}
    validation = [item for item in validation if item[1] not in train_keys]
    if not train or not validation:
        raise ValueError("grouped split has no training or validation rows after deduplication")
    return [row for row, _ in train], [row for row, _ in validation]


class TeacherDataset(Dataset):
    def __init__(self, rows: list[dict], target_transform: str, use_draw_state: bool = False):
        self.rows = rows
        self.target_transform = target_transform
        self.use_draw_state = use_draw_state
        if use_draw_state and any("repetition_twofold" not in row for row in rows):
            raise ValueError("--draw-state requires repetition_twofold on every record; FEN has no history")
        if use_draw_state and any(row["repetition_twofold"] not in (0, 1, False, True) for row in rows):
            raise ValueError("repetition_twofold must be a binary flag")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        board = chess.Board(row["fen"])
        cp = float(row["eval_cp"])
        target = 10 * math.tanh(cp / 1000) if self.target_transform == "existing_tanh" else cp / 100
        return board, target, row.get("repetition_twofold") if self.use_draw_state else None


def collate(rows):
    boards, targets, repetitions = zip(*rows)
    batch = boards_to_batch(list(boards))
    if repetitions[0] is not None:
        batch.draw_state[:, 1] = torch.tensor(repetitions, dtype=torch.float32)
    return batch, torch.tensor(targets, dtype=torch.float32)[:, None]


def evaluate(model: RayGNN, loader: DataLoader, device: str) -> float:
    model.eval()
    loss_sum, count = 0.0, 0
    with torch.inference_mode():
        for batch, target in loader:
            output = model(batch.to(device))
            target = target.to(device)
            loss_sum += nn.functional.huber_loss(output, target, delta=1.0, reduction="sum").item()
            count += target.numel()
    return loss_sum / count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="JSONL with fen, White-positive eval_cp and game_id")
    parser.add_argument("--output", type=Path, required=True, help="best checkpoint path")
    parser.add_argument("--variant", default="reference", choices=(
        "reference", "original_compact", "no_message", "raw_board_only", "channelwise_gates",
        "king_relative", "smaller_messages", "smaller_readout", "relation_biases",
        "one_layer", "three_layers"))
    parser.add_argument("--epochs", type=int, default=20, help="total epochs, usually 20 screen or 50 confirm")
    parser.add_argument("--horizon", type=int, default=50, help="schedule length, fixed from screening start")
    parser.add_argument("--epoch-size", type=int, default=1_000_000, help="sampled positions across all devices")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target-transform", choices=("existing_tanh", "cp_pawns"), default="existing_tanh")
    parser.add_argument("--augment-color", action="store_true")
    parser.add_argument("--draw-state", action="store_true", help="use halfmove/100 and twofold repetition")
    parser.add_argument("--float32-reductions", action="store_true")
    parser.add_argument("--edge-chunk-size", type=int)
    parser.add_argument("--sparse-board-projection", action="store_true")
    parser.add_argument("--resume", type=Path, help="resume this architecture's latest checkpoint")
    args = parser.parse_args()
    if min(args.epochs, args.horizon, args.epoch_size, args.batch_size) < 1 or args.epochs > args.horizon:
        parser.error("epochs, horizon, epoch-size and batch-size must be positive; epochs <= horizon")
    config = RayGNNConfig.variant(args.variant)
    config = replace(config, float32_reductions=args.float32_reductions,
                     edge_chunk_size=args.edge_chunk_size,
                     sparse_board_projection=args.sparse_board_projection,
                     draw_state_fields=("halfmove_clock_div_100", "repetition_twofold") if args.draw_state else ())
    torch.manual_seed(args.seed)
    training, validation = load_splits(args.data)
    train_set = TeacherDataset(training, args.target_transform, args.draw_state)
    val_set = TeacherDataset(validation, args.target_transform, args.draw_state)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=collate)
    model = RayGNN(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.horizon)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))
    best, first_epoch = float("inf"), 0
    if args.resume:
        saved = torch.load(args.resume, map_location=args.device, weights_only=True)
        if saved["config"] != asdict(config) or saved["seed"] != args.seed or saved["target_transform"] != args.target_transform or saved["horizon"] != args.horizon or saved["epoch_size"] != args.epoch_size or saved["augment_color"] != args.augment_color:
            raise ValueError("resume checkpoint experiment contract differs from this run")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        best, first_epoch = saved["best_validation"], saved["epoch"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    latest = args.output.with_name(args.output.stem + ".latest" + args.output.suffix)
    metadata = {
        "config": asdict(config), "seed": args.seed, "horizon": args.horizon,
        "epoch_size": args.epoch_size, "augment_color": args.augment_color,
        "target_transform": args.target_transform,
        "input_schema": {"piece": "square-major IDs 0..12", "side_to_move": "+1 White/-1 Black",
                         "castling": "WK,WQ,BK,BQ", "en_passant": "FEN target 0..63; 64 none",
                         "draw_state": list(config.draw_state_fields)},
        "edge_feature_order": ["relative_delta", "is_knight", "blocker_count_one_hot",
                               "first_blocker_piece_one_hot", "first_blocker_delta", "blocker_present",
                               "geometry_matches", "direct_attack_defense", "source_occupied",
                               "destination_occupied", "pawn_forward_geometry"],
        "perspective": "White-positive pawn units",
    }
    for epoch in range(first_epoch, args.epochs):
        sampler = RandomSampler(train_set, replacement=True, num_samples=args.epoch_size,
                                generator=torch.Generator().manual_seed(args.seed + epoch))
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, collate_fn=collate)
        augmentation_rng = torch.Generator().manual_seed(args.seed + 1_000_000 + epoch)
        model.train()
        loss_sum = 0.0
        for batch, target in train_loader:
            batch, target = batch.to(args.device), target.to(args.device)
            if args.augment_color:
                mask = (torch.rand(target.shape[0], generator=augmentation_rng) < 0.5).to(args.device)
                if bool(mask.any()):
                    transformed = swap_colors_reflect_ranks(batch)
                    batch.piece[mask] = transformed.piece[mask]
                    batch.side_to_move[mask] = transformed.side_to_move[mask]
                    batch.castling[mask] = transformed.castling[mask]
                    batch.en_passant[mask] = transformed.en_passant[mask]
                    target[mask] = -target[mask]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=args.device.startswith("cuda")):
                output = model(batch)
                loss = nn.functional.huber_loss(output, target, delta=1.0)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * target.shape[0]
        scheduler.step()
        validation_loss = evaluate(model, val_loader, args.device)
        print(f"epoch={epoch + 1} train={loss_sum / args.epoch_size:.5f} validation={validation_loss:.5f}")
        if validation_loss < best:
            best = validation_loss
            torch.save({**metadata, "model": model.state_dict(), "epoch": epoch + 1,
                        "best_validation": best}, args.output)
        torch.save({**metadata, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch + 1, "best_validation": best}, latest)


if __name__ == "__main__":
    main()
