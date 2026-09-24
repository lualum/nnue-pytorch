"""Synchronous contextual message passing over all candidate edges."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .geometry import BoardGeometry


class RayLayer(nn.Module):
    def __init__(self, message_width: int = 32, channelwise_gates: bool = False,
                 relation_biases: bool = False, float32_reductions: bool = False,
                 edge_chunk_size: int | None = None):
        super().__init__()
        self.message_width = message_width
        self.channelwise_gates = channelwise_gates
        self.float32_reductions = float32_reductions
        self.edge_chunk_size = edge_chunk_size
        self.source_project = nn.Linear(64, message_width, bias=False)
        self.target_project = nn.Linear(64, message_width, bias=False)
        self.state_project = nn.Linear(16, message_width, bias=False)
        if channelwise_gates:
            self.message_gate = nn.Linear(message_width, message_width)
            self.feedback_gate = nn.Linear(message_width, message_width)
            nn.init.zeros_(self.message_gate.bias)
            nn.init.zeros_(self.feedback_gate.bias)
        else:
            self.message_gate = nn.Linear(message_width, 1)
            self.feedback_gate = nn.Linear(message_width, 1)
            nn.init.zeros_(self.message_gate.bias)
            nn.init.zeros_(self.feedback_gate.bias)
        self.relation_message_bias = nn.Parameter(torch.zeros(3)) if relation_biases else None
        self.relation_feedback_bias = nn.Parameter(torch.zeros(3)) if relation_biases else None
        self.update = nn.Sequential(nn.Linear(64 + 2 * message_width + 16, 64),
                                    nn.SiLU(), nn.Linear(64, 64))
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, h: Tensor, state: Tensor, edges: Tensor, geometry: BoardGeometry,
                classes: Tensor | None = None) -> Tensor:
        b = h.shape[0]
        p, q, c = self.source_project(h), self.target_project(h), self.state_project(state)
        reduction_dtype = torch.float32 if self.float32_reductions else h.dtype
        incoming = torch.zeros((b, 64, self.message_width), device=h.device, dtype=reduction_dtype)
        outgoing = torch.zeros_like(incoming)
        chunk = self.edge_chunk_size or geometry.src.numel()
        for start in range(0, geometry.src.numel(), chunk):
            end = min(start + chunk, geometry.src.numel())
            src, dst = geometry.src[start:end], geometry.dst[start:end]
            x = F.silu(p[:, src] + q[:, dst] + edges[:, start:end] + c[:, None])
            if self.channelwise_gates:
                message_logit = self.message_gate(x)
                feedback_logit = self.feedback_gate(x)
            else:
                message_logit = self.message_gate(x)
                feedback_logit = self.feedback_gate(x)
            if self.relation_message_bias is not None:
                if classes is None:
                    raise ValueError("relation classes required for relation-specific biases")
                message_logit = message_logit + self.relation_message_bias[classes[:, start:end]][..., None]
                feedback_logit = feedback_logit + self.relation_feedback_bias[classes[:, start:end]][..., None]
            message = (torch.sigmoid(message_logit) * x).to(reduction_dtype)
            feedback = (torch.sigmoid(feedback_logit) * x).to(reduction_dtype)
            incoming.scatter_add_(1, dst[None, :, None].expand(b, -1, self.message_width), message)
            outgoing.scatter_add_(1, src[None, :, None].expand(b, -1, self.message_width), feedback)
        node_input = torch.cat((h, incoming.to(h.dtype), outgoing.to(h.dtype),
                                state[:, None].expand(-1, 64, -1)), -1)
        return h + self.alpha * self.update(node_input)
