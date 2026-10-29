"""Direct baseline model: frozen Whisper encoder + attentive-pooling intent head.

audio -> log-mel (Whisper processor) -> frozen Whisper encoder (no decoding)
      -> attentive pooling over valid frames -> MLP -> 60-way softmax
No transcript is produced at any point.
"""
import torch
import torch.nn as nn


class AttentivePoolingHead(nn.Module):
    def __init__(self, d_model: int, n_classes: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.score = nn.Linear(d_model, 1)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout),
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, h, mask):
        # h: (B, T, D) encoder states, mask: (B, T) True for valid frames
        s = self.score(h).squeeze(-1).masked_fill(~mask, float("-inf"))
        w = torch.softmax(s, dim=-1).unsqueeze(-1)
        pooled = (w * h).sum(1)
        return self.mlp(pooled)


@torch.no_grad()
def encode(encoder, proc, wav, device, dtype):
    """Return encoder states truncated to the frames that cover real audio."""
    from common import SR, n_encoder_frames
    feats = proc(wav, sampling_rate=SR, return_tensors="pt").input_features.to(device, dtype)
    h = encoder(feats).last_hidden_state[0]  # (1500, D)
    return h[: n_encoder_frames(len(wav))]
