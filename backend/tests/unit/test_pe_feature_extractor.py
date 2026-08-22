"""Unit test for PEFeatureExtractor."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from haruspex.forensics.pe_feature_extractor import PEFeatureExtractor


def test_pe_feature_extractor_synthetic_bytes():
    """Test feature extraction on synthetic binary bytes."""
    extractor = PEFeatureExtractor()
    # Construct a minimal DOS/PE header byte sequence
    dos_header = bytearray(64)
    dos_header[0:2] = b"MZ"
    dos_header[60:64] = (128).to_bytes(4, "little")  # e_lfanew

    pe_header = bytearray(256)
    pe_header[0:4] = b"PE\x00\x00"

    data = bytes(dos_header + pe_header + b"\x00" * 2000)

    features = extractor.extract(data)

    assert isinstance(features, np.ndarray)
    assert features.shape == (2381,)
    assert features.dtype == np.float32
    assert not np.isnan(features).any()

    # Byte histogram sum should be close to 1.0
    byte_hist = features[0:256]
    assert abs(byte_hist.sum() - 1.0) < 1e-4
