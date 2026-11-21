"""SI-MSET: subject-invariant multi-scale EOG Transformer."""
import torch
from torch import nn
import torch.nn.functional as functional

from eog_si_mset.protocol import variant_flags


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, weight):
        ctx.weight = weight
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.weight * gradient, None


class ScaleBranch(nn.Module):
    def __init__(self, width, kernel, dilation, dropout):
        super().__init__()
        padding = dilation * (kernel - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(width, width, kernel, padding=padding, dilation=dilation, groups=width, bias=False),
            nn.GroupNorm(8, width),
            nn.GELU(),
            nn.Conv1d(width, width, 1, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, values):
        return values + self.block(values)


class MultiScaleFrontend(nn.Module):
    def __init__(self, width, sequence_tokens, dropout, use_multiscale):
        super().__init__()
        self.projection = nn.Conv1d(3, width, 9, stride=4, padding=4, bias=False)
        self.use_multiscale = use_multiscale
        if use_multiscale:
            self.branches = nn.ModuleList([
                ScaleBranch(width, 7, 1, dropout),
                ScaleBranch(width, 15, 2, dropout),
                ScaleBranch(width, 31, 4, dropout),
            ])
            self.gate = nn.Sequential(
                nn.Conv1d(width * 3, width, 1),
                nn.Sigmoid(),
            )
        else:
            self.branches = nn.ModuleList([ScaleBranch(width, 15, 2, dropout)])
        self.pool_size = sequence_tokens

    def forward(self, values):
        values = self.projection(values)
        branch_values = [branch(values) for branch in self.branches]
        if self.use_multiscale:
            stacked = torch.cat(branch_values, dim=1)
            gate = self.gate(stacked)
            fused = gate * torch.stack(branch_values, dim=0).sum(dim=0)
        else:
            fused = branch_values[0]
        return functional.adaptive_avg_pool1d(fused, self.pool_size)


class AttentionPool(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.score = nn.Linear(width, 1)

    def forward(self, values):
        weights = torch.softmax(self.score(values).squeeze(-1), dim=1)
        return torch.sum(values * weights.unsqueeze(-1), dim=1)


class SIMSET(nn.Module):
    def __init__(self, n_subjects, variant="full", d_model=96, transformer_layers=3,
                 attention_heads=4, ffn_dim=192, sequence_tokens=96, dropout=0.15):
        super().__init__()
        flags = variant_flags(variant)
        self.variant = variant
        self.flags = flags
        self.frontend = MultiScaleFrontend(
            d_model, sequence_tokens, dropout, flags["use_multiscale"]
        )
        self.position = nn.Parameter(torch.randn(1, sequence_tokens, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=attention_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=transformer_layers, enable_nested_tensor=False
        )
        self.pool = AttentionPool(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.binary_head = nn.Linear(d_model, 1)
        self.kss_head = nn.Linear(d_model, 1)
        self.subject_head = nn.Linear(d_model, n_subjects) if flags["use_adversarial"] else None

    def forward(self, values, reversal_weight=0.0):
        tokens = self.frontend(values).transpose(1, 2)
        if tokens.shape[1] != self.position.shape[1]:
            raise ValueError("Frontend token count differs from positional embedding")
        tokens = self.encoder(tokens + self.position)
        embedding = self.norm(self.pool(tokens))
        output = {
            "binary_logit": self.binary_head(embedding).squeeze(-1),
            "kss": torch.sigmoid(self.kss_head(embedding).squeeze(-1)),
            "embedding": embedding,
            "subject_logits": None,
        }
        if self.subject_head is not None:
            reversed_embedding = GradientReversal.apply(embedding, reversal_weight)
            output["subject_logits"] = self.subject_head(reversed_embedding)
        return output
