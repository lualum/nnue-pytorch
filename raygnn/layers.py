"""Synchronous v3 relationship-separated message passing."""

import torch
from torch import Tensor, nn

from .geometry import BoardGeometry


class RayLayer(nn.Module):
    def __init__(self, message_width: int = 32, relation_separation: bool = True,
                 multiplicative_pair: bool = True, edge_chunk_size: int | None = None,
                 dense_reference: bool = False):
        super().__init__()
        self.message_width = message_width
        self.relation_separation = relation_separation
        self.multiplicative_pair = multiplicative_pair
        self.edge_chunk_size = edge_chunk_size
        self.dense_reference = dense_reference
        channels = 3 if relation_separation else 1
        pair_width = message_width * (4 if multiplicative_pair else 3)
        hidden = 96 if message_width == 48 else 64
        self.norm = nn.LayerNorm(96, eps=1e-5)
        self.source_project = nn.Linear(96, message_width, bias=False)
        self.target_project = nn.Linear(96, message_width, bias=False)
        self.state_project = nn.Linear(16, message_width, bias=False)
        self.pair = nn.ModuleList(nn.Sequential(nn.Linear(pair_width, hidden), nn.SiLU(),
                                                 nn.Linear(hidden, 2 * message_width))
                                  for _ in range(channels))
        update_width = 96 + 2 * channels * message_width + 2 * channels + 16
        self.update = nn.Sequential(nn.Linear(update_width, 192), nn.SiLU(), nn.Linear(192, 96))
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, h: Tensor, state: Tensor, edges: Tensor, geometry: BoardGeometry,
                classes: Tensor) -> Tensor:
        batch_size = h.shape[0]
        width = self.message_width
        channels = len(self.pair)
        n = self.norm(h)
        p = self.source_project(n) + self.state_project(state)[:, None]
        q = self.target_project(n)
        aggregates_in = torch.zeros((batch_size * 64 * channels, width), device=h.device, dtype=torch.float32)
        aggregates_out = torch.zeros_like(aggregates_in)
        counts_in = torch.zeros((batch_size * 64 * channels,), device=h.device, dtype=torch.float32)
        counts_out = torch.zeros_like(counts_in)
        # The dense reference visits all candidates; the packed path visits only active ones.
        candidates = torch.arange(classes.numel(), device=h.device) if self.dense_reference else torch.nonzero(classes.reshape(-1) >= 0).flatten()
        chunk_size = self.edge_chunk_size or max(1, candidates.numel())
        for selected in candidates.split(chunk_size):
            if selected.numel() == 0:
                continue
            b = selected // geometry.src.numel()
            e = selected % geometry.src.numel()
            relation = classes[b, e]
            if not self.dense_reference:
                active = relation >= 0
                b, e, relation = b[active], e[active], relation[active]
            if relation.numel() == 0:
                continue
            src, dst = geometry.src[e], geometry.dst[e]
            source, target = p[b, src], q[b, dst]
            parts = (source, target, source * target, edges[b, e]) if self.multiplicative_pair else (source, target, edges[b, e])
            pair_input = torch.cat(parts, -1)
            for r, mlp in enumerate(self.pair):
                mask = (relation >= 0) if channels == 1 else relation == r
                rb, rs, rd = b[mask], src[mask], dst[mask]
                outputs = mlp(pair_input if self.dense_reference else pair_input[mask])
                message, feedback = (outputs[mask] if self.dense_reference else outputs).split(width, -1)
                incoming_index = (rb * 64 + rd) * channels + r
                outgoing_index = (rb * 64 + rs) * channels + r
                aggregates_in.index_add_(0, incoming_index, message.float())
                aggregates_out.index_add_(0, outgoing_index, feedback.float())
                counts_in.index_add_(0, incoming_index, torch.ones_like(incoming_index, dtype=torch.float32))
                counts_out.index_add_(0, outgoing_index, torch.ones_like(outgoing_index, dtype=torch.float32))
        shape = (batch_size, 64, channels, width)
        incoming = (aggregates_in / counts_in.clamp_min(1).sqrt()[:, None]).reshape(shape).to(h.dtype)
        outgoing = (aggregates_out / counts_out.clamp_min(1).sqrt()[:, None]).reshape(shape).to(h.dtype)
        count_features = torch.cat((counts_in.reshape(batch_size, 64, channels).log1p(),
                                    counts_out.reshape(batch_size, 64, channels).log1p()), -1).to(h.dtype)
        node_input = torch.cat((n, incoming.flatten(2), outgoing.flatten(2), count_features,
                                state[:, None].expand(-1, 64, -1)), -1)
        return h + self.alpha * self.update(node_input)
