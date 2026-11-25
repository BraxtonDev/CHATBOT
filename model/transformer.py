
# LEM v3.0 - Decoder-only Transformer
# All weights are initialized randomly.

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        assert config["n_embd"] % config["n_head"] == 0

        self.n_head = config["n_head"]
        self.head_size = config["n_embd"] // config["n_head"]

        self.qkv = nn.Linear(config["n_embd"], 3 * config["n_embd"])
        self.proj = nn.Linear(config["n_embd"], config["n_embd"])

        self.attn_dropout = nn.Dropout(config["dropout"])
        self.resid_dropout = nn.Dropout(config["dropout"])

    def forward(self, x):

        B, T, C = x.shape

        q, k, v = self.qkv(x).split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.head_size ** 0.5)

        mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool),
            diagonal=1
        )

        scores = scores.masked_fill(mask, float("-inf"))

        attention = F.softmax(scores, dim=-1)
        attention = self.attn_dropout(attention)

        output = attention @ v

        output = output.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.proj(output))


class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(config["n_embd"], 4 * config["n_embd"]),
            nn.GELU(),
            nn.Linear(4 * config["n_embd"], config["n_embd"]),
            nn.Dropout(config["dropout"])
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config["n_embd"])
        self.attention = CausalSelfAttention(config)

        self.ln2 = nn.LayerNorm(config["n_embd"])
        self.feed_forward = FeedForward(config)

    def forward(self, x):

        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))

        return x


class LEMTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config["vocab_size"],
            config["n_embd"]
        )

        self.position_embedding = nn.Embedding(
            config["block_size"],
            config["n_embd"]
        )

        self.dropout = nn.Dropout(config["dropout"])

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(config)
                for _ in range(config["n_layer"])
            ]
        )

        self.final_norm = nn.LayerNorm(config["n_embd"])

        self.output_head = nn.Linear(
            config["n_embd"],
            config["vocab_size"],
            bias=False
        )

        # Weight sharing between input and output embeddings.
        self.output_head.weight = self.token_embedding.weight

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):

        B, T = idx.shape

        if T > self.config["block_size"]:
            raise ValueError("Input exceeds model context length")

        positions = torch.arange(T, device=idx.device)

        x = (
            self.token_embedding(idx)
            + self.position_embedding(positions)
        )

        x = self.dropout(x)
        x = self.blocks(x)
        x = self.final_norm(x)

        logits = self.output_head(x)

        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

        return logits, loss

