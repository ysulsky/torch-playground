import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


class MultiHeadLinear(nn.Module):
  """Applies multiple independent linear projections."""

  def __init__(
      self, n_heads: int, in_features: int, out_features: int, axis: int):
    super().__init__()
    self.axis = axis
    self.heads = nn.ModuleList([
      nn.Linear(in_features, out_features, bias=False)
      for _ in range(n_heads)
    ])

  def forward(self, x: Tensor) -> Tensor:
    outs = [head(x) for head in self.heads]
    return torch.stack(outs, dim=self.axis)


class AttentionBlock(nn.Module):
  """Attention Block."""

  @classmethod
  def causal_self_attention_mask(cls, seq_len: int) -> Tensor:
    return torch.tril(torch.ones((seq_len, seq_len), dtype=bool))

  def __init__(self, n_heads: int, d_model: int, dropout: float):
    super().__init__()
    if d_model % n_heads:
      raise ValueError(f"{d_model=} not divisibile by {n_heads=}")
    d_head = d_model // n_heads
    self.attn_prenorm = nn.LayerNorm(d_model)
    self.ffw_prenorm = nn.LayerNorm(d_model)
    self.kfn = MultiHeadLinear(n_heads, d_model, d_head, axis=-3)
    self.qfn = MultiHeadLinear(n_heads, d_model, d_head, axis=-3)
    self.vfn = MultiHeadLinear(n_heads, d_model, d_head, axis=-3)
    self.ffw = nn.Sequential(nn.Linear(d_model, d_model * 4),
                             nn.GELU(),
                             nn.Linear(d_model * 4, d_model))
    self.dropout = nn.Dropout(dropout)

  def forward(
      self, x: Tensor, y: Tensor | None = None,
      mask: Tensor | None = None) -> Tensor:
    B, Tx, D = x.shape
    x_norm = self.attn_prenorm(x)
    if y is None:
      y = x
      y_norm = x_norm
    else:
      y_norm = self.attn_prenorm(y)
    _, Ty, _ = y.shape
    assert B == y.shape[0] and D == y.shape[-1]
    q = self.qfn(x_norm)  # [B, H, Tx, D//H]
    k = self.kfn(y_norm)  # [B, H, Ty, D//H]
    v = self.vfn(y_norm)  # [B, H, Ty, D//H]
    logits = torch.einsum('...xd,...yd->...xy', q, k)  # [B, H, Tx, Ty]
    logits = logits * v.shape[-1] ** -0.5
    logits = self.dropout(logits)
    if mask is not None:
      assert mask.shape == (Tx, Ty)
      logits = torch.where(mask, logits, torch.tensor(float('-inf')))
    attn_out = torch.softmax(logits, dim=-1) @ v  # [B, H, Tx, D//H]
    attn_out = torch.transpose(attn_out, -3, -2)  # [B, Tx, H, D//H]
    attn_out = torch.reshape(attn_out, (B, Tx, D))
    attn_out = self.dropout(attn_out)
    ffw_input = self.ffw_prenorm(x + attn_out)
    return x + self.dropout(self.ffw(ffw_input))


class Transformer(nn.Module):
  """Decoder-only transformer."""

  def __init__(self, n_blocks: int, n_heads: int, seq_len: int, d_model: int,
               dropout: float):
    super().__init__()
    self.seq_len = seq_len
    self.d_model = d_model
    self.mask = self.register_buffer(
        'mask', AttentionBlock.causal_self_attention_mask(seq_len))
    self.position_embeddings = nn.Embedding(seq_len, d_model)
    self.blocks = nn.ModuleList([
        AttentionBlock(n_heads=n_heads, d_model=d_model, dropout=dropout)
        for _ in range(n_blocks)
    ])
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, sequence: Tensor) -> Tensor:
    B, T, D = sequence.shape
    assert self.seq_len == T
    assert self.d_model == D
    x = sequence + self.position_embeddings.weight
    x = self.dropout(x)
    for block in self.blocks:
      x = block(x, mask=self.mask)
    return x

  def loss(self, sequence: Tensor, target: Tensor) -> float:
    logits = self(sequence)
    assert logits.shape[:-1] == target.shape
    logits = torch.reshape(logits, (-1, self.d_model))
    target = torch.reshape(target, (-1,))
    return F.cross_entropy(logits, target)
