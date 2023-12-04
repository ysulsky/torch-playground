import unittest
import torch
from src import model


class MultiHeadLinearTest(unittest.TestCase):

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

  def test_self_attn_shape(self):
    batch_size = 2
    n_heads = 3
    d_model = 18
    seq_len = 6
    net = model.AttentionBlock(n_heads=n_heads, d_model=d_model,
                               dropout=0.5)
    mask = net.causal_self_attention_mask(seq_len)
    input = torch.zeros((batch_size, seq_len, d_model))
    output = net(input, mask=mask)
    self.assertEqual(input.shape, output.shape)

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
    output = net(input, context, mask=None)
    self.assertEqual(input.shape, output.shape)


class TransformerTest(unittest.TestCase):

  def test_shape(self):
    batch_size = 2
    n_heads = 3
    d_model = 18
    seq_len = 6

    seq = torch.randn((batch_size, seq_len, d_model))
    net = model.Transformer(n_blocks=2, n_heads=n_heads, seq_len=seq_len,
                            d_model=d_model, dropout=0.1)
    out = net(seq)
    self.assertEqual(seq.shape, out.shape)
    target = torch.zeros((batch_size, seq_len), dtype=torch.long)
    loss = net.loss(seq, target)
    self.assertEqual(loss.shape, ())



if __name__ == "__main__":
  unittest.main()