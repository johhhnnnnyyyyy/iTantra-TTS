import numpy as np
import pytest

from tools.audit_indic_tts_quantization import (
    fold_weight_norm,
    relative_rmse,
    simulated_int4_roundtrip,
)


def test_fold_weight_norm_matches_dim_zero_convolution() -> None:
    v = np.array([[[3.0, 4.0]], [[6.0, 8.0]]], dtype=np.float32)
    g = np.array([5.0, 20.0], dtype=np.float32)
    np.testing.assert_allclose(fold_weight_norm(v, g), np.array([[[3, 4]], [[12, 16]]]))


def test_int4_roundtrip_stays_on_signed_four_bit_grid() -> None:
    weights = np.array([[0.0, 0.4, 1.0, -1.0]], dtype=np.float32)
    rounded = simulated_int4_roundtrip(weights, block_size=4)
    np.testing.assert_allclose(rounded, np.rint(weights * 7) / 7)
    assert relative_rmse(weights, rounded) > 0


def test_audit_math_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="zero-norm"):
        fold_weight_norm(np.zeros((1, 2, 3), dtype=np.float32), np.ones(1))
    with pytest.raises(ValueError, match="positive"):
        simulated_int4_roundtrip(np.ones((1, 2), dtype=np.float32), block_size=0)
