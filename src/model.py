import dataclasses
import torch
import torch.distributions as distributions
import torch.nn as nn

Categorical = distributions.Categorical
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


@dataclasses.dataclass
class KVCache:
  seq_len: int
  size: int
  k_cache: torch.Tensor
  v_cache: torch.Tensor

  @classmethod
  def empty(
      cls, batch_size: int, seq_len: int,
      n_heads: int, d_model: int) -> 'KVCache':
    d_head = d_model // n_heads
    k_cache = v_cache = torch.zeros((batch_size, n_heads, seq_len, d_head))
    return cls(seq_len=seq_len, size=0, k_cache=k_cache, v_cache=v_cache)

  def to_drop_for_update(self, new_els: int) -> tuple[int, int]:
    return max(0, self.size + new_els - self.seq_len)

  def update(self, new_k: Tensor, new_v: Tensor) -> 'KVCache':
    batch_size, n_heads, new_els, d_head = new_k.shape
    to_drop = self.to_drop_for_update(new_els)
    remaining = self.size - to_drop
    pad_amt = self.seq_len - (remaining + new_els)
    padding = torch.zeros((batch_size, n_heads, pad_amt, d_head))
    return KVCache(
        seq_len=self.seq_len,
        size=remaining + new_els,
        k_cache=torch.cat(
            (self.k_cache[:, :, to_drop : to_drop + remaining, :],
             new_k, padding), dim=-2),
        v_cache=torch.cat(
            (self.v_cache[:, :, to_drop : to_drop + remaining, :],
             new_v, padding), dim=-2),
    )


class AttentionBlock(nn.Module):
  """Attention Block."""

  def __init__(self, n_heads: int, d_model: int, dropout: float):
    super().__init__()
    if d_model % n_heads:
      raise ValueError(f"{d_model=} not divisibile by {n_heads=}")
    d_head = d_model // n_heads
    self.attn_prenorm = lambda x:x #nn.LayerNorm(d_model)
    self.ffw_prenorm = lambda x:x #nn.LayerNorm(d_model)
    self.kfn = MultiHeadLinear(n_heads, d_model, d_head, axis=-3)
    self.qfn = MultiHeadLinear(n_heads, d_model, d_head, axis=-3)
    self.vfn = MultiHeadLinear(n_heads, d_model, d_head, axis=-3)
    self.ffw = nn.Sequential(nn.Linear(d_model, d_model * 4),
                             nn.GELU(),
                             nn.Linear(d_model * 4, d_model))
    self.dropout = lambda x: x#nn.Dropout(dropout)

  @classmethod
  def causal_self_attention_mask(cls, seq_len: int) -> Tensor:
    return torch.tril(torch.ones((seq_len, seq_len), dtype=bool))

  def forward(
      self, x: Tensor, y: Tensor | None = None,
      mask: Tensor | None = None,
      cache: KVCache | None = None) -> tuple[Tensor, KVCache | None]:
    B, Tx, D = x.shape
    x_norm = self.attn_prenorm(x)
    if y is None:
      y = x
      y_norm = x_norm
    else:
      y_norm = self.attn_prenorm(y)
    Ty = y.shape[1] if cache is None else cache.seq_len
    assert B == y.shape[0] and D == y.shape[-1]
    q = self.qfn(x_norm)  # [B, H, Tx, D//H]466,  0.2759, -0.6625],
    k = self.kfn(y_norm)  # [B, H, Ty, D//H]
    v = self.vfn(y_norm)  # [B, H, Ty, D//H]
    if cache is not None:
      cache = cache.update(k, v)
      k, v = cache.k_cache, cache.v_cache
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
    return x + self.dropout(self.ffw(ffw_input)), cache


class Transformer(nn.Module):
  """Decoder-only transformer."""

  def __init__(self, n_blocks: int, n_heads: int, seq_len: int, d_model: int,
               vocab_size: int, dropout: float):
    super().__init__()
    self.seq_len = seq_len
    self.d_model = d_model
    self.n_heads = n_heads
    self.vocab_size = vocab_size
    self.register_buffer(
        'ffw_mask', AttentionBlock.causal_self_attention_mask(seq_len))
    self.token_embeddings = nn.Embedding(vocab_size, d_model)
    self.position_embeddings = nn.Embedding(seq_len, d_model)
    self.blocks = nn.ModuleList([
        AttentionBlock(n_heads=n_heads, d_model=d_model, dropout=dropout)
        for _ in range(n_blocks)
    ])
    self.proj = nn.Linear(d_model, vocab_size)
    self.dropout = nn.Dropout(p=dropout)

  def initialize_cache(self, batch_size: int) -> list[KVCache]:
    return [KVCache.empty(batch_size=batch_size,
                          seq_len=self.seq_len,
                          n_heads=self.n_heads,
                          d_model=self.d_model)] * len(self.blocks)

  def forward(self, tokens: Tensor,
              cache: list[KVCache] | None = None
      ) -> tuple[Tensor, list[KVCache] | None]:
    x = self.token_embeddings(tokens)
    if cache is None:
      x = x + self.position_embeddings.weight
      mask = self.ffw_mask
    else:
      num_toks = tokens.shape[-1]
      pos_start = cache[0].size - cache[0].to_drop_for_update(num_toks)
      pos_end = pos_start + num_toks
      x = x + self.position_embeddings.weight[pos_start : pos_end]
      mask = self.ffw_mask[pos_start : pos_end]
    x = self.dropout(x)
    if cache is not None:
      in_cache = cache
    else:
      in_cache = [None] * len(self.blocks)
    out_cache = [None] * len(self.blocks)
    for i, block in enumerate(self.blocks):
      x, out_cache[i] = block(x, mask=mask, cache=in_cache[i])
    if cache is not None:
      cache = out_cache
    return self.proj(x), cache

  def loss(self, tokens: Tensor, target: Tensor) -> float:
    logits, _ = self(tokens, None)
    assert logits.shape == (*target.shape, self.vocab_size)
    logits = torch.reshape(logits, (-1, self.vocab_size))
    target = torch.reshape(target, (-1,))
    return Categorical(logits=logits).log_prob(target).mean()

  def sample(self, prefix: Tensor) -> Tensor:
    batch_size, prefix_len = prefix.shape
    logits, cache = self(prefix, self.initialize_cache(batch_size))
    samples = [Categorical(logits=logits[:, -1]).sample()]
    while len(samples) + prefix_len < self.seq_len:
      logits, cache = self(torch.unsqueeze(samples[-1], 1), cache=cache)
      samples.append(Categorical(logits=logits[:, -1]).sample())
    return torch.stack(samples, dim=1)
