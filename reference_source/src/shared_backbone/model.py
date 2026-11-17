"""Model components for the shared-backbone DeltaGate scenario.

The intended surface is:

- one shared SASRec-style sequence encoder;
- one item embedding/output head per domain;
- a target-domain DeltaGate adapter that can freeze the shared encoder.

This module is deliberately small. Training, negative sampling, and evaluation
belong in separate scripts so the shared/frozen parameter contract stays easy to
audit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class SharedSASRecEncoder(nn.Module):
    """SASRec-style Transformer body shared across domains."""

    def __init__(
        self,
        embedding_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        max_seq_len: int = 50,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("position_ids", torch.arange(max_seq_len).unsqueeze(0), persistent=False)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones((max_seq_len, max_seq_len), dtype=torch.bool), diagonal=1),
            persistent=False,
        )
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.01)

    def forward(self, item_embeddings: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Return the final sequence state for left-padded item embeddings.

        Args:
            item_embeddings: Tensor shaped [batch, seq_len, embedding_dim].
            padding_mask: Boolean tensor shaped [batch, seq_len], where True is padding.
        """
        batch_size, seq_len, _ = item_embeddings.shape
        positions = self.position_ids[:, :seq_len].expand(batch_size, -1)
        x = self.dropout(item_embeddings + self.position_embedding(positions))
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        x = self.layer_norm(x)
        return x[:, -1, :]


class SharedGRUEncoder(nn.Module):
    """GRU sequence encoder with the same input/output contract as SASRec."""

    def __init__(
        self,
        embedding_dim: int = 64,
        num_layers: int = 2,
        max_seq_len: int = 50,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, item_embeddings: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Return the last non-padding state for left-padded item embeddings."""
        x = self.dropout(item_embeddings)
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        output, _ = self.gru(x)
        return self.layer_norm(output[:, -1, :])


class DomainItemHead(nn.Module):
    """Domain-specific item embedding and dot-product output head."""

    def __init__(self, item_num: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.item_num = item_num
        self.embedding_dim = embedding_dim
        self.item_embedding = nn.Embedding(item_num + 1, embedding_dim, padding_idx=0)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def embed_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        return self.item_embedding(seq)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.matmul(hidden, self.item_embedding.weight[1:].T)


class SharedBackboneDeltaGateAdapter(nn.Module):
    """Target-domain DeltaGate adapter over a shared encoder and domain head."""

    def __init__(
        self,
        encoder: nn.Module,
        domain_head: DomainItemHead,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.domain_head = domain_head
        self.embedding_dim = embedding_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(1 + embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid(),
        )
        self.global_prior = nn.Parameter(torch.zeros(embedding_dim))
        nn.init.zeros_(self.global_prior)

    def forward(self, seq: torch.Tensor, delta_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        padding_mask = seq.eq(0)
        seq_emb = self.domain_head.embed_sequence(seq)
        h_personalized = self.encoder(seq_emb, padding_mask).detach()
        gate_input = torch.cat([torch.log1p(delta_t).unsqueeze(-1), h_personalized], dim=-1)
        gate = self.gate_mlp(gate_input)
        h_final = gate * h_personalized + (1.0 - gate) * self.global_prior.unsqueeze(0)
        return self.domain_head.logits(h_final), gate.mean(dim=-1)


@dataclass(frozen=True)
class ParameterReport:
    trainable: int
    total: int
    trainable_names: list[str]


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def trainable_parameter_report(module: nn.Module) -> ParameterReport:
    total = 0
    trainable = 0
    trainable_names = []
    for name, parameter in module.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
            trainable_names.append(name)
    return ParameterReport(trainable=trainable, total=total, trainable_names=trainable_names)
