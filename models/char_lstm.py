"""
models/char_lstm.py
--------------------
Character-level LSTM for text FL benchmarks (Shakespeare, Sent140).

Architecture
------------
    Embedding(vocab_size=80, embed_dim=8)
    LSTM(8 -> 256, num_layers=2, dropout=0.2)
    Linear(256 -> num_classes)

Parameters:  ~820K
Serialized:  ~3.3 MB

This matches the model used in the original FedAvg paper
(McMahan et al. 2017) for the Shakespeare next-character-prediction task.

Input
-----
    x: LongTensor of shape (batch, seq_len)  -- character indices
Output
------
    logits: FloatTensor of shape (batch, num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CharLSTM(nn.Module):
    """
    Two-layer character LSTM matching McMahan et al. 2017 (FedAvg paper).

    Parameters
    ----------
    num_classes : int
        Vocabulary size / number of output characters.
    vocab_size  : int
        Input vocabulary size (default 80 covers printable ASCII).
    embed_dim   : int
        Character embedding dimension.
    hidden_dim  : int
        LSTM hidden state size.
    num_layers  : int
        Number of stacked LSTM layers.
    dropout     : float
        Dropout probability between LSTM layers (only if num_layers > 1).
    """

    def __init__(
        self,
        num_classes: int = 80,
        vocab_size: int = 80,
        embed_dim: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x      : LongTensor (batch, seq_len)
        hidden : optional (h_0, c_0) tuple for stateful inference

        Returns
        -------
        logits : FloatTensor (batch, num_classes)
            Prediction at the last time step.
        """
        emb = self.embed(x)                    # (B, T, embed_dim)
        out, _ = self.lstm(emb, hidden)        # (B, T, hidden_dim)
        logits = self.fc(out[:, -1, :])        # last time step -> (B, num_classes)
        return logits


if __name__ == "__main__":
    import io
    model = CharLSTM(num_classes=80)
    total = sum(p.numel() for p in model.parameters())
    print(f"CharLSTM parameters: {total:,}")
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    print(f"Serialized size: {buf.tell() / 1e6:.2f} MB")
    dummy = torch.randint(0, 80, (4, 80))   # batch=4, seq_len=80
    out = model(dummy)
    print(f"Output shape: {out.shape}")     # (4, 80)
