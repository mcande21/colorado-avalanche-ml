"""Tests for PersistentSlabTransformer model architecture."""
from __future__ import annotations

import torch

from avalanche_ml.models.transformer_model import (
    PersistentSlabTransformer,
    PositionalEncoding,
)


def test_positional_encoding_shape():
    pe = PositionalEncoding(d_model=64, max_len=365)
    x = torch.randn(4, 14, 64)
    out = pe(x)
    assert out.shape == (4, 14, 64)


def test_positional_encoding_adds_to_input():
    pe = PositionalEncoding(d_model=64, max_len=365)
    x = torch.zeros(2, 7, 64)
    out = pe(x)
    assert out.abs().sum() > 0


def test_transformer_forward_shape():
    model = PersistentSlabTransformer(input_size=99, d_model=64, nhead=4, num_layers=2)
    x = torch.randn(8, 14, 99)
    out = model(x)
    assert out.shape == (8, 1)


def test_transformer_output_range():
    model = PersistentSlabTransformer(input_size=99, d_model=64, nhead=4, num_layers=2)
    x = torch.randn(16, 7, 99)
    out = model(x)
    assert (out >= 0).all() and (out <= 1).all()


def test_transformer_with_padding_mask():
    model = PersistentSlabTransformer(input_size=99, d_model=64, nhead=4, num_layers=2)
    x = torch.randn(4, 14, 99)
    mask = torch.zeros(4, 14, dtype=torch.bool)
    mask[0, 10:] = True
    mask[1, 12:] = True
    out = model(x, mask=mask)
    assert out.shape == (4, 1)
    assert (out >= 0).all() and (out <= 1).all()


def test_transformer_different_lookbacks():
    model = PersistentSlabTransformer(input_size=99, d_model=64, nhead=4, num_layers=2)
    for seq_len in [7, 14, 30]:
        x = torch.randn(4, seq_len, 99)
        out = model(x)
        assert out.shape == (4, 1)
