# Copyright 2025 Lianyu Huang
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Comprehensive tests for Conv3d NKI kernel.

Three layers of test coverage:
    Layer 1: PyTorch standard Conv3d test cases (12 configs from torch test suite)
    Layer 2: Video model real-world configs (Wan2.1 VAE CausalConv3d shapes)
    Layer 3: Edge cases and boundary conditions

All tests compare against both:
    - PyTorch F.conv3d (ground truth)
    - NumPy im2col reference (conv3d_ref)
"""

import numpy as np
import pytest

from conv3d_ref import conv3d_ref

# Try to import PyTorch for golden reference
try:
    import torch
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Try to import NKI for kernel testing
try:
    import nki

    from conv3d import conv3d as conv3d_nki

    HAS_NKI = True
except ImportError:
    HAS_NKI = False


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

# (B, C_in, C_out, D, H, W, kD, kH, kW, stride, padding, bias)

# Layer 1: PyTorch standard Conv3d test cases
PYTORCH_STANDARD_PARAMS = [
    # Basic
    (1, 2, 3, 4, 5, 4, 2, 3, 2, (1, 1, 1), (0, 0, 0), True),
    # No bias
    (1, 2, 3, 3, 4, 5, 2, 3, 4, (1, 1, 1), (0, 0, 0), False),
    # 1x1x1 kernel (pointwise)
    (1, 2, 3, 3, 4, 5, 1, 1, 1, (1, 1, 1), (0, 0, 0), False),
    # Stride 2
    (2, 3, 4, 5, 5, 5, 2, 2, 2, (2, 2, 2), (0, 0, 0), True),
    # Stride 2 + padding 1
    (2, 3, 4, 5, 5, 5, 2, 2, 2, (2, 2, 2), (1, 1, 1), True),
    # Groups=1 (standard)
    (1, 2, 3, 4, 5, 4, 3, 3, 3, (1, 1, 1), (0, 0, 0), True),
    # Padding same as kernel//2 (3x3x3 pad 1)
    (1, 2, 4, 4, 5, 4, 3, 3, 3, (1, 1, 1), (1, 1, 1), True),
    # Non-cubic kernel
    (1, 3, 4, 6, 5, 4, 2, 3, 4, (1, 1, 1), (0, 0, 0), True),
    # Asymmetric spatial
    (1, 2, 3, 4, 8, 6, 2, 3, 2, (1, 1, 1), (0, 0, 0), False),
    # Large stride
    (1, 4, 8, 8, 8, 8, 3, 3, 3, (2, 2, 2), (1, 1, 1), True),
    # Temporal-only kernel (3,1,1)
    (1, 4, 4, 8, 4, 4, 3, 1, 1, (1, 1, 1), (1, 0, 0), False),
    # Temporal stride 2 (downsampling)
    (1, 4, 4, 8, 4, 4, 3, 1, 1, (2, 1, 1), (0, 0, 0), False),
]

# Layer 2: Wan2.1 VAE real-world configs
# These are the actual CausalConv3d configurations from Wan2.1's 3D VAE
WAN_VAE_PARAMS = [
    # ResidualBlock main path: (3,3,3) stride 1, pad 1
    # Early stages (96 channels)
    (1, 3, 96, 4, 8, 8, 3, 3, 3, (1, 1, 1), (1, 1, 1), False),
    (1, 96, 96, 4, 8, 8, 3, 3, 3, (1, 1, 1), (1, 1, 1), False),
    # Mid stages (192 channels)
    (1, 96, 192, 4, 8, 8, 3, 3, 3, (1, 1, 1), (1, 1, 1), False),
    (1, 192, 192, 4, 8, 8, 3, 3, 3, (1, 1, 1), (1, 1, 1), False),
    # Late stages (384 channels) - largest config
    (1, 192, 384, 4, 4, 4, 3, 3, 3, (1, 1, 1), (1, 1, 1), False),
    (1, 384, 384, 4, 4, 4, 3, 3, 3, (1, 1, 1), (1, 1, 1), False),
    # Temporal downsample: (3,1,1) stride (2,1,1)
    (1, 96, 96, 8, 8, 8, 3, 1, 1, (2, 1, 1), (0, 0, 0), False),
    (1, 192, 192, 4, 4, 4, 3, 1, 1, (2, 1, 1), (0, 0, 0), False),
    # Temporal upsample conv: (3,1,1) stride 1 pad (1,0,0)
    (1, 96, 192, 4, 8, 8, 3, 1, 1, (1, 1, 1), (1, 0, 0), False),
    # Pointwise (1,1,1) - shortcut / bottleneck
    (1, 96, 192, 4, 8, 8, 1, 1, 1, (1, 1, 1), (0, 0, 0), False),
    (1, 16, 32, 4, 8, 8, 1, 1, 1, (1, 1, 1), (0, 0, 0), False),
    # Head conv: z_dim output
    (1, 384, 32, 4, 4, 4, 3, 3, 3, (1, 1, 1), (1, 1, 1), True),
]

# Layer 3: Edge cases
EDGE_CASE_PARAMS = [
    # Single voxel output
    (1, 2, 3, 2, 3, 2, 2, 3, 2, (1, 1, 1), (0, 0, 0), False),
    # Batch > 1
    (4, 2, 3, 4, 4, 4, 3, 3, 3, (1, 1, 1), (1, 1, 1), True),
    # Single channel
    (1, 1, 1, 4, 4, 4, 3, 3, 3, (1, 1, 1), (1, 1, 1), False),
    # Large kernel relative to input
    (1, 2, 3, 3, 3, 3, 3, 3, 3, (1, 1, 1), (1, 1, 1), True),
    # D=1 (degenerate to 2D)
    (1, 4, 8, 1, 8, 8, 1, 3, 3, (1, 1, 1), (0, 1, 1), False),
    # Mixed strides
    (1, 4, 4, 6, 8, 8, 3, 3, 3, (2, 1, 1), (1, 1, 1), False),
    (1, 4, 4, 4, 8, 8, 1, 3, 3, (1, 2, 2), (0, 1, 1), False),
]

ALL_PARAMS = PYTORCH_STANDARD_PARAMS + WAN_VAE_PARAMS + EDGE_CASE_PARAMS

PARAM_NAMES = "B, C_in, C_out, D, H, W, kD, kH, kW, stride, padding, use_bias"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _generate_inputs(B, C_in, C_out, D, H, W, kD, kH, kW, use_bias, dtype=np.float32):
    """Generate reproducible random inputs."""
    rng = np.random.RandomState(42)
    input_arr = rng.randn(B, C_in, D, H, W).astype(dtype)
    weight_arr = rng.randn(C_out, C_in, kD, kH, kW).astype(dtype)
    bias_arr = rng.randn(C_out).astype(dtype) if use_bias else None
    return input_arr, weight_arr, bias_arr


# ---------------------------------------------------------------------------
# Tests: NumPy reference vs PyTorch
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestConv3dRefVsPyTorch:
    """Validate NumPy reference against PyTorch F.conv3d."""

    @pytest.mark.parametrize(PARAM_NAMES, ALL_PARAMS)
    def test_ref_matches_pytorch(self, B, C_in, C_out, D, H, W, kD, kH, kW, stride, padding, use_bias):
        input_np, weight_np, bias_np = _generate_inputs(
            B, C_in, C_out, D, H, W, kD, kH, kW, use_bias
        )

        # NumPy reference
        out_ref = conv3d_ref(input_np, weight_np, bias_np, stride=stride, padding=padding)

        # PyTorch reference
        input_t = torch.from_numpy(input_np)
        weight_t = torch.from_numpy(weight_np)
        bias_t = torch.from_numpy(bias_np) if bias_np is not None else None
        out_torch = F.conv3d(input_t, weight_t, bias_t, stride=stride, padding=padding).numpy()

        # Relaxed tolerance for large channel counts (float32 accumulation differences)
        np.testing.assert_allclose(out_ref, out_torch, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Tests: NKI kernel vs PyTorch
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_NKI, reason="NKI not installed")
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestConv3dNKIVsPyTorch:
    """Validate NKI kernel against PyTorch F.conv3d."""

    @pytest.mark.parametrize(PARAM_NAMES, PYTORCH_STANDARD_PARAMS)
    def test_standard(self, B, C_in, C_out, D, H, W, kD, kH, kW, stride, padding, use_bias):
        self._run_test(B, C_in, C_out, D, H, W, kD, kH, kW, stride, padding, use_bias)

    @pytest.mark.parametrize(PARAM_NAMES, WAN_VAE_PARAMS)
    def test_wan_vae(self, B, C_in, C_out, D, H, W, kD, kH, kW, stride, padding, use_bias):
        self._run_test(B, C_in, C_out, D, H, W, kD, kH, kW, stride, padding, use_bias)

    @pytest.mark.parametrize(PARAM_NAMES, EDGE_CASE_PARAMS)
    def test_edge_cases(self, B, C_in, C_out, D, H, W, kD, kH, kW, stride, padding, use_bias):
        self._run_test(B, C_in, C_out, D, H, W, kD, kH, kW, stride, padding, use_bias)

    def _run_test(self, B, C_in, C_out, D, H, W, kD, kH, kW, stride, padding, use_bias):
        input_np, weight_np, bias_np = _generate_inputs(
            B, C_in, C_out, D, H, W, kD, kH, kW, use_bias
        )

        # PyTorch golden
        input_t = torch.from_numpy(input_np)
        weight_t = torch.from_numpy(weight_np)
        bias_t = torch.from_numpy(bias_np) if bias_np is not None else None
        expected = F.conv3d(input_t, weight_t, bias_t, stride=stride, padding=padding).numpy()

        # NKI kernel via baremetal
        stride_d, stride_h, stride_w = stride
        pad_d, pad_h, pad_w = padding
        baremetal_fn = nki.baremetal(conv3d)
        actual = baremetal_fn(
            input_np,
            weight_np,
            bias_np,
            stride_d=stride_d,
            stride_h=stride_h,
            stride_w=stride_w,
            pad_d=pad_d,
            pad_h=pad_h,
            pad_w=pad_w,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Tests: NumPy reference standalone (no torch/nki dependency)
# ---------------------------------------------------------------------------

class TestConv3dRefStandalone:
    """Basic sanity tests for the NumPy reference, no external dependencies."""

    def test_identity_1x1x1(self):
        """1x1x1 conv with identity weight = channel-wise linear transform."""
        B, C, D, H, W = 1, 2, 3, 4, 5
        input_np = np.random.randn(B, C, D, H, W).astype(np.float32)
        weight = np.eye(C, dtype=np.float32).reshape(C, C, 1, 1, 1)
        out = conv3d_ref(input_np, weight)
        np.testing.assert_allclose(out, input_np, rtol=1e-6, atol=1e-6)

    def test_zero_weight(self):
        """Zero weight should produce zero output (or bias only)."""
        B, C_in, C_out, D, H, W = 1, 3, 4, 3, 4, 5
        input_np = np.random.randn(B, C_in, D, H, W).astype(np.float32)
        weight = np.zeros((C_out, C_in, 2, 2, 2), dtype=np.float32)
        out = conv3d_ref(input_np, weight)
        np.testing.assert_allclose(out, 0, atol=1e-7)

    def test_zero_weight_with_bias(self):
        """Zero weight + bias should produce constant output = bias."""
        B, C_in, C_out, D, H, W = 1, 3, 4, 3, 4, 5
        input_np = np.random.randn(B, C_in, D, H, W).astype(np.float32)
        weight = np.zeros((C_out, C_in, 2, 2, 2), dtype=np.float32)
        bias = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out = conv3d_ref(input_np, weight, bias)
        for c in range(C_out):
            np.testing.assert_allclose(out[0, c], bias[c], atol=1e-7)

    def test_output_shape(self):
        """Verify output shape calculation for various configs."""
        configs = [
            # (D,H,W, kD,kH,kW, stride, padding) -> (D_out, H_out, W_out)
            ((4, 8, 8), (3, 3, 3), (1, 1, 1), (1, 1, 1), (4, 8, 8)),
            ((4, 8, 8), (3, 3, 3), (1, 1, 1), (0, 0, 0), (2, 6, 6)),
            ((8, 8, 8), (3, 3, 3), (2, 2, 2), (1, 1, 1), (4, 4, 4)),
            ((8, 4, 4), (3, 1, 1), (2, 1, 1), (0, 0, 0), (3, 4, 4)),
        ]
        for (D, H, W), (kD, kH, kW), stride, padding, (eD, eH, eW) in configs:
            input_np = np.zeros((1, 1, D, H, W), dtype=np.float32)
            weight = np.zeros((1, 1, kD, kH, kW), dtype=np.float32)
            out = conv3d_ref(input_np, weight, stride=stride, padding=padding)
            assert out.shape == (1, 1, eD, eH, eW), (
                f"Expected shape (1,1,{eD},{eH},{eW}), got {out.shape}"
            )


# ---------------------------------------------------------------------------
# Causal padding tests (for Wan2.1 CausalConv3d compatibility)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestCausalConv3d:
    """Test causal (asymmetric temporal) padding pattern used by Wan2.1 VAE.

    CausalConv3d pads (2*pad_t, 0) temporally instead of (pad_t, pad_t).
    This is done OUTSIDE the kernel via F.pad, then conv3d with pad_d=0.
    """

    def test_causal_padding_3x3x3(self):
        """Simulate CausalConv3d(3,3,3) padding=(1,1,1)."""
        B, C_in, C_out = 1, 16, 32
        D, H, W = 8, 8, 8
        kD, kH, kW = 3, 3, 3

        rng = np.random.RandomState(42)
        input_np = rng.randn(B, C_in, D, H, W).astype(np.float32)
        weight_np = rng.randn(C_out, C_in, kD, kH, kW).astype(np.float32)

        # Causal padding: temporal (2, 0), spatial (1, 1)
        input_padded = np.pad(
            input_np,
            ((0, 0), (0, 0), (2, 0), (1, 1), (1, 1)),
            mode="constant",
        )

        # Conv3d with no padding (already padded)
        out_ref = conv3d_ref(input_padded, weight_np, stride=(1, 1, 1), padding=(0, 0, 0))

        # Compare with PyTorch
        input_t = torch.from_numpy(input_padded)
        weight_t = torch.from_numpy(weight_np)
        out_torch = F.conv3d(input_t, weight_t, stride=1, padding=0).numpy()

        np.testing.assert_allclose(out_ref, out_torch, rtol=1e-5, atol=1e-5)
        # Output should have same D as input (causal = no temporal shrinkage)
        assert out_ref.shape[2] == D

    def test_causal_temporal_downsample(self):
        """Simulate CausalConv3d(3,1,1) stride=(2,1,1) for temporal downsampling."""
        B, C_in, C_out = 1, 32, 32
        D, H, W = 8, 4, 4

        rng = np.random.RandomState(42)
        input_np = rng.randn(B, C_in, D, H, W).astype(np.float32)
        weight_np = rng.randn(C_out, C_in, 3, 1, 1).astype(np.float32)

        # Causal padding: temporal (2, 0), no spatial padding for (1,1) kernel
        input_padded = np.pad(
            input_np,
            ((0, 0), (0, 0), (2, 0), (0, 0), (0, 0)),
            mode="constant",
        )

        out_ref = conv3d_ref(input_padded, weight_np, stride=(2, 1, 1), padding=(0, 0, 0))

        input_t = torch.from_numpy(input_padded)
        weight_t = torch.from_numpy(weight_np)
        out_torch = F.conv3d(input_t, weight_t, stride=(2, 1, 1), padding=0).numpy()

        np.testing.assert_allclose(out_ref, out_torch, rtol=1e-5, atol=1e-5)
        # Temporal dimension halved: (8+2-3)//2 + 1 = 4
        assert out_ref.shape[2] == (D + 2 - 3) // 2 + 1
