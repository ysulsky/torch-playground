"""Tests for model code."""

import unittest
import torch
import torch.testing
from src import model


class MultiHeadLinearTest(unittest.TestCase):

  @torch.no_grad()
  def test_shape(self):
    batch_size = 2
    n_heads = 3
    in_features = 4
    out_features = 5
    seq_len = 6
    net = model.MultiHeadLinear(n_heads=n_heads, in_features=in_features,
                                out_features=out_features, axis=-3)
    input = torch.zeros((batch_size, seq_len, in_features))
    out = net(input)
    self.assertEqual(out.shape, (batch_size, n_heads, seq_len, out_features))


class AttentionBlockTest(unittest.TestCase):

  @torch.no_grad()
  def test_self_attn_shape(self):
    batch_size = 2
    n_heads = 3
    d_model = 18
    seq_len = 6
    net = model.AttentionBlock(n_heads=n_heads, d_model=d_model,
                               dropout=0.5)
    mask = net.causal_self_attention_mask(seq_len)
    input = torch.zeros((batch_size, seq_len, d_model))
    output, _ = net(input, mask=mask)
    self.assertEqual(input.shape, output.shape)

  @torch.no_grad()
  def test_cross_attn_shape(self):
    batch_size = 2
    n_heads = 3
    d_model = 18
    seq_len = 6
    enc_seq_len = 7
    net = model.AttentionBlock(n_heads=n_heads, d_model=d_model,
                               dropout=0.5)
    input = torch.zeros((batch_size, seq_len, d_model))
    context = torch.zeros((batch_size, enc_seq_len, d_model))
    output, _ = net(input, context, mask=None)
    self.assertEqual(input.shape, output.shape)


class TransformerTest(unittest.TestCase):

  @torch.no_grad()
  def test_shape(self):
    batch_size = 2
    n_heads = 3
    d_model = 18
    vocab_size = 24
    seq_len = 2

    seq = torch.randint(vocab_size, (batch_size, seq_len))
    net = model.Transformer(n_blocks=2, n_heads=n_heads, seq_len=seq_len,
                            d_model=d_model, vocab_size=vocab_size,
                            dropout=0.1)
    out, _ = net(seq)
    self.assertEqual(out.shape, (batch_size, seq_len, vocab_size))
    target = (vocab_size - 1) * torch.ones(
      (batch_size, seq_len), dtype=torch.long)
    loss = net.loss(seq, target)
    self.assertEqual(loss.shape, ())

  @torch.no_grad()
  def test_cache_same_as_ffw(self):
    n_blocks = 2
    batch_size = 2
    n_heads = 3
    d_model = 18
    vocab_size = 24
    seq_len = 24

    seq = torch.randint(vocab_size, (batch_size, seq_len))
    net = model.Transformer(n_blocks=n_blocks, n_heads=n_heads, seq_len=seq_len,
                            d_model=d_model, vocab_size=vocab_size,
                            dropout=0.0)
    out_ffw, _ = net(seq)
    out_loop = [None] * seq_len
    cache = net.initialize_cache(batch_size)

    for t in range(seq_len):
      out_loop[t], cache = net(seq[:, t:t+1], cache=cache)
    torch.testing.assert_allclose(out_ffw, torch.cat(out_loop, dim=1),
                                  atol=1e-2, rtol=1e-2)

  @torch.no_grad()
  def test_sample_shape(self):
    n_blocks = 2
    batch_size = 2
    n_heads = 3
    d_model = 18
    vocab_size = 24
    seq_len = 24
    prefix_len = 10

    net = model.Transformer(n_blocks=n_blocks, n_heads=n_heads, seq_len=seq_len,
                            d_model=d_model, vocab_size=vocab_size,
                            dropout=0.1)
    prefix = torch.randint(vocab_size, (batch_size, prefix_len))
    suffix = net.sample(prefix)
    self.assertEqual(suffix.shape, (batch_size, seq_len - prefix_len))


if __name__ == "__main__":
  unittest.main()