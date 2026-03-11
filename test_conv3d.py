# Copyright 2026 Lianyu Huang
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Comprehensive tests for Conv3d NKI kernel.

Three layers of test coverage:
    Layer 1: PyTorch standard Conv3d test cases (12 configs from torch test suite)
    Layer 2: Video model real-world configs:
        - Wan2.1/2.2 VAE CausalConv3d shapes
        - CogVideoX-5b VAE (block_out=[128,256,256,512], latent=16)
        - HunyuanVideo VAE (block_out=[128,256,512,512], latent=16)
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
    import neuronxcc.nki as nki

    from conv3d import conv3d as conv3d_nki
    from conv3d import conv3d_kernel

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

# Layer 2: Wan2.1/2.2 VAE real-world configs
# These are the actual CausalConv3d configurations from Wan2.1/2.2's 3D VAE
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

# Layer 2b: CogVideoX VAE real-world configs (CogVideoX-5b)
# Source: HuggingFace THUDM/CogVideoX-5b vae/config.json
# Architecture: AutoencoderKLCogVideoX
#   block_out_channels: [128, 256, 256, 512], latent_channels: 16, layers_per_block: 3
#   All CausalConv3d use kernel_size=3; causal temporal padding = (kernel-1, 0)
#   Spatial downsampling via Conv2d stride 2 (not tested here - 2D op)
#   Temporal downsampling via avg_pool1d stride 2 (not tested here - 1D op)
#   Shortcut uses Conv3d kernel=1 when in_channels != out_channels
# (B, C_in, C_out, D, H, W, kD, kH, kW, stride, padding, bias)
COGVIDEOX_VAE_PARAMS = [
    # Encoder conv_in: RGB input -> first block channels
    (1, 3, 128, 4, 8, 8, 3, 3, 3, (1, 1, 1), (1, 1, 1), True),
    # Encoder block 0 ResNet: 128 -> 128 (3 layers per block)
    (1, 128, 128, 4, 8, 8, 3, 3, 3, (1, 1, 1), (1, 1, 1), False),
    # Encoder block 1 ResNet: 128 -> 256 (first layer, channel change)
    (1, 128, 256, 4, 4, 4, 3, 3, 3, (1, 1, 1), (1, 1, 1), False),
    # Encoder block 1 ResNet: 256 -> 256
    (1, 256, 256, 4, 4, 4, 3, 3, 3, (1, 1, 1), (1, 1, 1), False),
    # Encoder block 1 shortcut: 128 -> 256, 1x1x1 pointwise
    (1, 128, 256, 4, 4, 4, 1, 1, 1, (1, 1, 1), (0, 0, 0), False),
    # Encoder block 2 ResNet: 256 -> 256 (no channel change)
    (1, 256, 256, 2, 2, 2, 3, 3, 3, (1, 1, 1), (1, 1, 1), False),
    # Encoder block 3 ResNet: 256 -> 512 (first layer, channel change)
    (1, 256, 512, 2, 1, 1, 3, 3, 3, (1, 1, 1), (1, 1, 1), False),
    # Encoder block 3 ResNet: 512 -> 512
    (1, 512, 512, 2, 1, 1, 3, 3, 3, (1, 1, 1), (1, 1, 1), False),
    # Encoder block 3 shortcut: 256 -> 512, 1x1x1 pointwise
    (1, 256, 512, 2, 1, 1, 1, 1, 1, (1, 1, 1), (0, 0, 0), False),
    # Encoder mid-block ResNet: 512 -> 512
    (1, 512, 512, 2, 1, 1, 3, 3, 3, (1, 1, 1), (1, 1, 1), False),
    # Encoder conv_out: last block -> 2 * latent_channels (mean + logvar)
    (1, 512, 32, 2, 1, 1, 3, 3, 3, (1, 1, 1), (1, 1, 1), True),
    # Decoder conv_in: latent_channels -> last block
    (1, 16, 512, 2, 1, 1, 3, 3, 3, (1, 1, 1), (1, 1, 1), True),
    # Decoder block (reversed) ResNet: 512 -> 256 (4 layers per block in decoder)
    (1, 512, 256, 2, 2, 2, 3, 3, 3, (1, 1, 1), (1, 1, 1), False),
    # Decoder block ResNet: 256 -> 128
    (1, 256, 128, 4, 4, 4, 3, 3, 3, (1, 1, 1), (1, 1, 1), False),
    # Decoder conv_out: first block -> RGB
    (1, 128, 3, 4, 8, 8, 3, 3, 3, (1, 1, 1), (1, 1, 1), True),
]

# Layer 2c: HunyuanVideo VAE real-world configs
# Source: HuggingFace tencent/HunyuanVideo hunyuan-video-t2v-720p/vae/config.json
# Architecture: AutoencoderKLCausal3D
#   block_out_channels: [128, 256, 512, 512], latent_channels: 16, layers_per_block: 2
#   CausalConv3d: kernel=3, causal padding = (kernel-1, 0, pad_h, pad_h, pad_w, pad_w)
#   Spatial+temporal downsample: CausalConv3d kernel=3 stride=2
#   Shortcut uses CausalConv3d kernel=1 when in_channels != out_channels
# (B, C_in, C_out, D, H, W, kD, kH, kW, stride, padding, bias)
HUNYUANVIDEO_VAE_PARAMS = [
    # Encoder conv_in: RGB input -> first block channels
    (1, 3, 128, 4, 8, 8, 3, 3, 3, (1, 1, 1), (1, 1, 1), True),
    # Encoder block 0 ResNet: 128 -> 128 (2 layers per block)
    (1, 128, 128, 4, 8, 8, 3, 3, 3, (1, 1, 1), (1, 1, 1), False),
    # Encoder block 0 downsample: CausalConv3d kernel=3 stride=2
    (1, 128, 128, 4, 8, 8, 3, 3, 3, (2, 2, 2), (1, 1, 1), False),
    # Encoder block 1 ResNet: 128 -> 256 (first layer, channel change)
    (1, 128, 256, 2, 4, 4, 3, 3, 3, (1, 1, 1), (1, 1, 1), False),
    # Encoder block 1 ResNet: 256 -> 256
    (1, 256, 256, 2, 4, 4, 3, 3, 3, (1, 1, 1), (1, 1, 1), False),
    # Encoder block 1 shortcut: 128 -> 256, 1x1x1
    (1, 128, 256, 2, 4, 4, 1, 1, 1, (1, 1, 1), (0, 0, 0), False),
    # Encoder block 1 downsample: stride 2
    (1, 256, 256, 2, 4, 4, 3, 3, 3, (2, 2, 2), (1, 1, 1), False),
    # Encoder block 2 ResNet: 256 -> 512 (channel change)
    (1, 256, 512, 1, 2, 2, 3, 3, 3, (1, 1, 1), (1, 1, 1), False),
    # Encoder block 2 ResNet: 512 -> 512
    (1, 512, 512, 1, 2, 2, 3, 3, 3, (1, 1, 1), (1, 1, 1), False),
    # Encoder block 2 shortcut: 256 -> 512
    (1, 256, 512, 1, 2, 2, 1, 1, 1, (1, 1, 1), (0, 0, 0), False),
    # Encoder block 2 downsample: stride 2
    (1, 512, 512, 1, 2, 2, 3, 3, 3, (2, 2, 2), (1, 1, 1), False),
    # Encoder block 3 ResNet: 512 -> 512 (no channel change)
    (1, 512, 512, 1, 1, 1, 3, 3, 3, (1, 1, 1), (1, 1, 1), False),
    # Encoder mid-block: 512 -> 512
    (1, 512, 512, 1, 1, 1, 3, 3, 3, (1, 1, 1), (1, 1, 1), False),
    # Encoder conv_out: 512 -> 32 (2 * latent_channels for mean + logvar)
    (1, 512, 32, 1, 1, 1, 3, 3, 3, (1, 1, 1), (1, 1, 1), True),
    # Decoder conv_in: latent -> 512
    (1, 16, 512, 1, 1, 1, 3, 3, 3, (1, 1, 1), (1, 1, 1), True),
    # Decoder block ResNet: 512 -> 256
    (1, 512, 256, 2, 4, 4, 3, 3, 3, (1, 1, 1), (1, 1, 1), False),
    # Decoder block ResNet: 256 -> 128
    (1, 256, 128, 4, 8, 8, 3, 3, 3, (1, 1, 1), (1, 1, 1), False),
    # Decoder conv_out: 128 -> RGB
    (1, 128, 3, 4, 8, 8, 3, 3, 3, (1, 1, 1), (1, 1, 1), True),
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

ALL_PARAMS = (PYTORCH_STANDARD_PARAMS + WAN_VAE_PARAMS
              + COGVIDEOX_VAE_PARAMS + HUNYUANVIDEO_VAE_PARAMS
              + EDGE_CASE_PARAMS)

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

    @pytest.mark.parametrize(PARAM_NAMES, COGVIDEOX_VAE_PARAMS)
    def test_cogvideox_vae(self, B, C_in, C_out, D, H, W, kD, kH, kW, stride, padding, use_bias):
        self._run_test(B, C_in, C_out, D, H, W, kD, kH, kW, stride, padding, use_bias)

    @pytest.mark.parametrize(PARAM_NAMES, HUNYUANVIDEO_VAE_PARAMS)
    def test_hunyuanvideo_vae(self, B, C_in, C_out, D, H, W, kD, kH, kW, stride, padding, use_bias):
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

        # NKI kernel via conv3d wrapper (pads input, then calls kernel)
        actual = conv3d_nki(
            input_np, weight_np, bias_np,
            stride=stride, padding=padding,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Tests: CogVideoX VAE Conv3d shapes (ref vs PyTorch)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestConv3dCogVideoX:
    """Validate Conv3d with CogVideoX-5b VAE layer configurations.

    CogVideoX uses AutoencoderKLCogVideoX with CausalConv3d (kernel=3).
    Config: block_out_channels=[128, 256, 256, 512], latent_channels=16,
    layers_per_block=3. Spatial downsampling via Conv2d stride 2;
    temporal downsampling via avg_pool1d stride 2.

    Source: HuggingFace THUDM/CogVideoX-5b vae/config.json
    Diffusers: autoencoder_kl_cogvideox.py
    """

    @pytest.mark.parametrize(PARAM_NAMES, COGVIDEOX_VAE_PARAMS)
    def test_ref_matches_pytorch(self, B, C_in, C_out, D, H, W, kD, kH, kW,
                                  stride, padding, use_bias):
        """NumPy reference vs PyTorch for CogVideoX VAE shapes."""
        input_np, weight_np, bias_np = _generate_inputs(
            B, C_in, C_out, D, H, W, kD, kH, kW, use_bias
        )

        out_ref = conv3d_ref(input_np, weight_np, bias_np, stride=stride, padding=padding)

        input_t = torch.from_numpy(input_np)
        weight_t = torch.from_numpy(weight_np)
        bias_t = torch.from_numpy(bias_np) if bias_np is not None else None
        out_torch = F.conv3d(input_t, weight_t, bias_t, stride=stride, padding=padding).numpy()

        np.testing.assert_allclose(out_ref, out_torch, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Tests: HunyuanVideo VAE Conv3d shapes (ref vs PyTorch)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestConv3dHunyuanVideo:
    """Validate Conv3d with HunyuanVideo VAE layer configurations.

    HunyuanVideo uses AutoencoderKLCausal3D with CausalConv3d (kernel=3).
    Config: block_out_channels=[128, 256, 512, 512], latent_channels=16,
    layers_per_block=2. Spatial+temporal downsample via CausalConv3d
    kernel=3 stride=2 (unlike CogVideoX which uses Conv2d + avg_pool).

    Source: HuggingFace tencent/HunyuanVideo hunyuan-video-t2v-720p/vae/config.json
    Code: hyvideo/vae/unet_causal_3d_blocks.py
    """

    @pytest.mark.parametrize(PARAM_NAMES, HUNYUANVIDEO_VAE_PARAMS)
    def test_ref_matches_pytorch(self, B, C_in, C_out, D, H, W, kD, kH, kW,
                                  stride, padding, use_bias):
        """NumPy reference vs PyTorch for HunyuanVideo VAE shapes."""
        input_np, weight_np, bias_np = _generate_inputs(
            B, C_in, C_out, D, H, W, kD, kH, kW, use_bias
        )

        out_ref = conv3d_ref(input_np, weight_np, bias_np, stride=stride, padding=padding)

        input_t = torch.from_numpy(input_np)
        weight_t = torch.from_numpy(weight_np)
        bias_t = torch.from_numpy(bias_np) if bias_np is not None else None
        out_torch = F.conv3d(input_t, weight_t, bias_t, stride=stride, padding=padding).numpy()

        np.testing.assert_allclose(out_ref, out_torch, rtol=1e-4, atol=1e-4)


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
    """Test causal (asymmetric temporal) padding pattern used by Wan2.1/2.2 VAE.

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


# ---------------------------------------------------------------------------
# BFloat16 precision tests
# ---------------------------------------------------------------------------

def _to_bf16_precision(arr_f32):
    """Quantize a float32 numpy array to bfloat16 precision via torch round-trip.

    NumPy does not natively support bfloat16, so we round-trip through
    torch.bfloat16 to truncate mantissa bits, then convert back to float32.
    This gives us float32 arrays whose values are exactly representable in bf16.
    """
    t = torch.from_numpy(arr_f32).to(torch.bfloat16).to(torch.float32)
    return t.numpy()


def _generate_bf16_inputs(B, C_in, C_out, D, H, W, kD, kH, kW, use_bias):
    """Generate random inputs quantized to bfloat16 precision (stored as float32)."""
    rng = np.random.RandomState(42)
    input_arr = _to_bf16_precision(rng.randn(B, C_in, D, H, W).astype(np.float32))
    weight_arr = _to_bf16_precision(rng.randn(C_out, C_in, kD, kH, kW).astype(np.float32))
    bias_arr = _to_bf16_precision(rng.randn(C_out).astype(np.float32)) if use_bias else None
    return input_arr, weight_arr, bias_arr


# BFloat16 test parameters (subset - smaller configs to keep tests fast)
BF16_BASIC_PARAMS = [
    # Basic small conv3d
    (1, 2, 3, 4, 5, 4, 2, 3, 2, (1, 1, 1), (0, 0, 0), False),
    # 1x1x1 pointwise
    (1, 4, 8, 3, 4, 5, 1, 1, 1, (1, 1, 1), (0, 0, 0), False),
    # With padding
    (1, 2, 4, 4, 5, 4, 3, 3, 3, (1, 1, 1), (1, 1, 1), False),
    # With stride
    (2, 3, 4, 5, 5, 5, 2, 2, 2, (2, 2, 2), (0, 0, 0), False),
    # With bias
    (1, 2, 3, 4, 5, 4, 2, 3, 2, (1, 1, 1), (0, 0, 0), True),
    # Non-cubic kernel
    (1, 3, 4, 6, 5, 4, 2, 3, 4, (1, 1, 1), (0, 0, 0), True),
]

BF16_WAN_VAE_PARAMS = [
    # Wan2.1/2.2 VAE first layer: 3 -> 128 channels, 3x3x3 kernel
    (1, 3, 128, 4, 8, 8, 3, 3, 3, (1, 1, 1), (1, 1, 1), False),
    # Wan2.1/2.2 VAE first layer with bias
    (1, 3, 128, 4, 8, 8, 3, 3, 3, (1, 1, 1), (1, 1, 1), True),
    # Temporal downsample
    (1, 96, 96, 8, 8, 8, 3, 1, 1, (2, 1, 1), (0, 0, 0), False),
    # Pointwise shortcut
    (1, 96, 192, 4, 8, 8, 1, 1, 1, (1, 1, 1), (0, 0, 0), False),
]


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestConv3dBFloat16:
    """Test conv3d with bfloat16-precision inputs and weights.

    Since numpy does not support bfloat16 natively, we quantize float32 values
    to bf16 precision via torch round-trip. The NKI kernel receives these
    bf16-precision float32 arrays. We compare against PyTorch F.conv3d run
    in bfloat16 (cast back to float32 for comparison).

    Tolerances are relaxed (atol=1e-2) to account for bf16 reduced mantissa.
    """

    @pytest.mark.parametrize(PARAM_NAMES, BF16_BASIC_PARAMS)
    def test_bf16_basic(self, B, C_in, C_out, D, H, W, kD, kH, kW, stride, padding, use_bias):
        """Basic conv3d cases with bfloat16-precision data."""
        self._run_bf16_ref_test(B, C_in, C_out, D, H, W, kD, kH, kW, stride, padding, use_bias)

    @pytest.mark.parametrize(PARAM_NAMES, BF16_WAN_VAE_PARAMS)
    def test_bf16_wan_vae(self, B, C_in, C_out, D, H, W, kD, kH, kW, stride, padding, use_bias):
        """Wan2.1/2.2 VAE configs with bfloat16-precision data."""
        self._run_bf16_ref_test(B, C_in, C_out, D, H, W, kD, kH, kW, stride, padding, use_bias)

    def test_bf16_with_bias(self):
        """Dedicated test for bias addition in bfloat16 precision."""
        B, C_in, C_out = 1, 4, 8
        D, H, W = 4, 5, 5
        kD, kH, kW = 3, 3, 3
        stride = (1, 1, 1)
        padding = (1, 1, 1)

        input_np, weight_np, bias_np = _generate_bf16_inputs(
            B, C_in, C_out, D, H, W, kD, kH, kW, use_bias=True
        )

        # NumPy reference with bf16-precision inputs
        out_ref = conv3d_ref(input_np, weight_np, bias_np, stride=stride, padding=padding)

        # PyTorch bf16 golden reference
        input_t = torch.from_numpy(input_np).to(torch.bfloat16)
        weight_t = torch.from_numpy(weight_np).to(torch.bfloat16)
        bias_t = torch.from_numpy(bias_np).to(torch.bfloat16)
        out_torch = F.conv3d(input_t, weight_t, bias_t, stride=stride, padding=padding)
        out_torch = out_torch.to(torch.float32).numpy()

        np.testing.assert_allclose(out_ref, out_torch, rtol=5e-2, atol=1e-2)

    def test_bf16_precision_acceptable(self):
        """Verify that bf16 output error vs float32 is within expected bounds.

        bf16 has ~7 bits of mantissa (vs 23 for fp32), so relative error
        should be roughly 2^-7 ~ 0.008 for well-conditioned operations.
        """
        B, C_in, C_out = 1, 4, 8
        D, H, W = 4, 6, 6
        kD, kH, kW = 3, 3, 3
        stride = (1, 1, 1)
        padding = (1, 1, 1)

        rng = np.random.RandomState(42)
        input_f32 = rng.randn(B, C_in, D, H, W).astype(np.float32)
        weight_f32 = rng.randn(C_out, C_in, kD, kH, kW).astype(np.float32)

        # Full float32 reference
        input_t32 = torch.from_numpy(input_f32)
        weight_t32 = torch.from_numpy(weight_f32)
        out_f32 = F.conv3d(input_t32, weight_t32, stride=stride, padding=padding)

        # bfloat16 computation
        input_bf16 = input_t32.to(torch.bfloat16)
        weight_bf16 = weight_t32.to(torch.bfloat16)
        out_bf16 = F.conv3d(input_bf16, weight_bf16, stride=stride, padding=padding)

        # Measure relative error
        out_f32_np = out_f32.numpy()
        out_bf16_np = out_bf16.to(torch.float32).numpy()

        abs_diff = np.abs(out_f32_np - out_bf16_np)
        scale = np.maximum(np.abs(out_f32_np), 1e-6)
        rel_error = np.mean(abs_diff / scale)

        # bf16 relative error should be < 5% for typical conv3d
        assert rel_error < 0.05, (
            f"bf16 mean relative error {rel_error:.4f} exceeds 5% threshold"
        )

    def _run_bf16_ref_test(self, B, C_in, C_out, D, H, W, kD, kH, kW, stride, padding, use_bias):
        """Compare NumPy reference (bf16-precision f32) against PyTorch bf16."""
        input_np, weight_np, bias_np = _generate_bf16_inputs(
            B, C_in, C_out, D, H, W, kD, kH, kW, use_bias
        )

        # NumPy reference with bf16-precision float32 arrays
        out_ref = conv3d_ref(input_np, weight_np, bias_np, stride=stride, padding=padding)

        # PyTorch bf16 golden
        input_t = torch.from_numpy(input_np).to(torch.bfloat16)
        weight_t = torch.from_numpy(weight_np).to(torch.bfloat16)
        bias_t = torch.from_numpy(bias_np).to(torch.bfloat16) if bias_np is not None else None
        out_torch = F.conv3d(input_t, weight_t, bias_t, stride=stride, padding=padding)
        out_torch = out_torch.to(torch.float32).numpy()

        np.testing.assert_allclose(out_ref, out_torch, rtol=5e-2, atol=1e-2)


@pytest.mark.skipif(not HAS_NKI, reason="NKI not installed")
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestConv3dNKIBFloat16:
    """Test NKI kernel with bfloat16-precision inputs against PyTorch bf16."""

    @pytest.mark.parametrize(PARAM_NAMES, BF16_BASIC_PARAMS)
    def test_nki_bf16_basic(self, B, C_in, C_out, D, H, W, kD, kH, kW, stride, padding, use_bias):
        """NKI kernel with bf16-precision data vs PyTorch bf16."""
        self._run_nki_bf16_test(B, C_in, C_out, D, H, W, kD, kH, kW, stride, padding, use_bias)

    @pytest.mark.parametrize(PARAM_NAMES, BF16_WAN_VAE_PARAMS)
    def test_nki_bf16_wan_vae(self, B, C_in, C_out, D, H, W, kD, kH, kW, stride, padding, use_bias):
        """NKI kernel Wan VAE configs with bf16-precision data."""
        self._run_nki_bf16_test(B, C_in, C_out, D, H, W, kD, kH, kW, stride, padding, use_bias)

    def test_nki_bf16_with_bias(self):
        """NKI kernel bias addition in bfloat16 precision."""
        B, C_in, C_out = 1, 4, 8
        D, H, W = 4, 5, 5
        kD, kH, kW = 3, 3, 3
        stride = (1, 1, 1)
        padding = (1, 1, 1)

        input_np, weight_np, bias_np = _generate_bf16_inputs(
            B, C_in, C_out, D, H, W, kD, kH, kW, use_bias=True
        )

        # PyTorch bf16 golden
        input_t = torch.from_numpy(input_np).to(torch.bfloat16)
        weight_t = torch.from_numpy(weight_np).to(torch.bfloat16)
        bias_t = torch.from_numpy(bias_np).to(torch.bfloat16)
        expected = F.conv3d(input_t, weight_t, bias_t, stride=stride, padding=padding)
        expected = expected.to(torch.float32).numpy()

        # NKI kernel with bf16-precision float32 arrays
        actual = conv3d_nki(input_np, weight_np, bias_np, stride=stride, padding=padding)

        np.testing.assert_allclose(actual, expected, rtol=5e-2, atol=1e-2)

    def _run_nki_bf16_test(self, B, C_in, C_out, D, H, W, kD, kH, kW, stride, padding, use_bias):
        """Compare NKI kernel (bf16-precision f32) against PyTorch bf16."""
        input_np, weight_np, bias_np = _generate_bf16_inputs(
            B, C_in, C_out, D, H, W, kD, kH, kW, use_bias
        )

        # PyTorch bf16 golden
        input_t = torch.from_numpy(input_np).to(torch.bfloat16)
        weight_t = torch.from_numpy(weight_np).to(torch.bfloat16)
        bias_t = torch.from_numpy(bias_np).to(torch.bfloat16) if bias_np is not None else None
        expected = F.conv3d(input_t, weight_t, bias_t, stride=stride, padding=padding)
        expected = expected.to(torch.float32).numpy()

        # NKI kernel with bf16-precision float32 arrays
        actual = conv3d_nki(input_np, weight_np, bias_np, stride=stride, padding=padding)

        np.testing.assert_allclose(actual, expected, rtol=5e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Dilation tests
# ---------------------------------------------------------------------------

# (B, C_in, C_out, D, H, W, kD, kH, kW, stride, padding, dilation, bias)
DILATION_PARAMS = [
    # Uniform dilation (2,2,2) with small input
    (1, 2, 3, 6, 8, 8, 2, 3, 3, (1, 1, 1), (0, 0, 0), (2, 2, 2), False),
    # Spatial-only dilation (1,2,2)
    (1, 3, 4, 4, 8, 8, 2, 3, 3, (1, 1, 1), (0, 0, 0), (1, 2, 2), True),
    # Temporal-only dilation (2,1,1)
    (1, 4, 4, 8, 5, 5, 3, 2, 2, (1, 1, 1), (0, 0, 0), (2, 1, 1), False),
    # Dilation with padding
    (1, 2, 3, 6, 8, 8, 3, 3, 3, (1, 1, 1), (2, 2, 2), (2, 2, 2), True),
    # Dilation with stride
    (1, 3, 4, 8, 10, 10, 2, 3, 3, (2, 2, 2), (0, 0, 0), (2, 2, 2), False),
    # Asymmetric dilation
    (1, 2, 3, 8, 10, 10, 2, 3, 3, (1, 1, 1), (0, 0, 0), (2, 3, 1), False),
]

DILATION_PARAM_NAMES = "B, C_in, C_out, D, H, W, kD, kH, kW, stride, padding, dilation, use_bias"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestConv3dDilation:
    """Test conv3d dilation support against PyTorch F.conv3d."""

    @pytest.mark.parametrize(DILATION_PARAM_NAMES, DILATION_PARAMS)
    def test_ref_dilation_vs_pytorch(self, B, C_in, C_out, D, H, W, kD, kH, kW,
                                     stride, padding, dilation, use_bias):
        """Validate NumPy reference with dilation against PyTorch."""
        input_np, weight_np, bias_np = _generate_inputs(
            B, C_in, C_out, D, H, W, kD, kH, kW, use_bias
        )

        out_ref = conv3d_ref(input_np, weight_np, bias_np,
                             stride=stride, padding=padding, dilation=dilation)

        input_t = torch.from_numpy(input_np)
        weight_t = torch.from_numpy(weight_np)
        bias_t = torch.from_numpy(bias_np) if bias_np is not None else None
        out_torch = F.conv3d(input_t, weight_t, bias_t,
                             stride=stride, padding=padding, dilation=dilation).numpy()

        np.testing.assert_allclose(out_ref, out_torch, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(DILATION_PARAM_NAMES, DILATION_PARAMS)
    @pytest.mark.skipif(not HAS_NKI, reason="NKI not installed")
    def test_nki_dilation_vs_pytorch(self, B, C_in, C_out, D, H, W, kD, kH, kW,
                                     stride, padding, dilation, use_bias):
        """Validate NKI conv3d with dilation against PyTorch."""
        input_np, weight_np, bias_np = _generate_inputs(
            B, C_in, C_out, D, H, W, kD, kH, kW, use_bias
        )

        input_t = torch.from_numpy(input_np)
        weight_t = torch.from_numpy(weight_np)
        bias_t = torch.from_numpy(bias_np) if bias_np is not None else None
        expected = F.conv3d(input_t, weight_t, bias_t,
                            stride=stride, padding=padding, dilation=dilation).numpy()

        actual = conv3d_nki(input_np, weight_np, bias_np,
                            stride=stride, padding=padding, dilation=dilation)

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    def test_dilation_1_matches_no_dilation(self):
        """Dilation=(1,1,1) should produce identical results to default (no dilation)."""
        B, C_in, C_out = 1, 3, 4
        D, H, W = 4, 6, 6
        kD, kH, kW = 2, 3, 3

        input_np, weight_np, bias_np = _generate_inputs(
            B, C_in, C_out, D, H, W, kD, kH, kW, use_bias=True
        )

        out_default = conv3d_ref(input_np, weight_np, bias_np,
                                 stride=(1, 1, 1), padding=(0, 0, 0))
        out_dilation1 = conv3d_ref(input_np, weight_np, bias_np,
                                   stride=(1, 1, 1), padding=(0, 0, 0),
                                   dilation=(1, 1, 1))

        np.testing.assert_allclose(out_default, out_dilation1, rtol=1e-7, atol=1e-7)

    def test_dilation_output_shape(self):
        """Verify output shape with dilation matches expected formula."""
        # dilation=2, kernel=3 -> effective kernel = 5
        # input spatial=10, no padding -> output = (10 - 5) // 1 + 1 = 6
        B, C_in, C_out = 1, 1, 1
        D, H, W = 10, 10, 10
        kD, kH, kW = 3, 3, 3

        input_np = np.zeros((B, C_in, D, H, W), dtype=np.float32)
        weight_np = np.zeros((C_out, C_in, kD, kH, kW), dtype=np.float32)

        out = conv3d_ref(input_np, weight_np,
                         stride=(1, 1, 1), padding=(0, 0, 0),
                         dilation=(2, 2, 2))

        assert out.shape == (1, 1, 6, 6, 6), f"Expected (1,1,6,6,6), got {out.shape}"


# ---------------------------------------------------------------------------
# Grouped convolution tests
# ---------------------------------------------------------------------------

def _generate_grouped_inputs(B, C_in, C_out, D, H, W, kD, kH, kW, groups, use_bias,
                             dtype=np.float32):
    """Generate reproducible random inputs for grouped convolution.

    Weight shape is [C_out, C_in/groups, kD, kH, kW] to match PyTorch convention.
    """
    rng = np.random.RandomState(42)
    C_in_per_group = C_in // groups
    input_arr = rng.randn(B, C_in, D, H, W).astype(dtype)
    weight_arr = rng.randn(C_out, C_in_per_group, kD, kH, kW).astype(dtype)
    bias_arr = rng.randn(C_out).astype(dtype) if use_bias else None
    return input_arr, weight_arr, bias_arr


# (B, C_in, C_out, D, H, W, kD, kH, kW, stride, padding, groups, bias)
GROUPED_CONV_PARAMS = [
    # groups=2, basic
    (1, 4, 6, 4, 5, 4, 2, 3, 2, (1, 1, 1), (0, 0, 0), 2, False),
    # groups=2, with bias
    (1, 4, 6, 4, 5, 4, 2, 3, 2, (1, 1, 1), (0, 0, 0), 2, True),
    # groups=2, with padding
    (1, 4, 8, 4, 5, 5, 3, 3, 3, (1, 1, 1), (1, 1, 1), 2, True),
    # groups=2, with stride
    (2, 4, 6, 6, 6, 6, 2, 2, 2, (2, 2, 2), (0, 0, 0), 2, False),
    # groups=4
    (1, 8, 12, 4, 5, 5, 2, 3, 3, (1, 1, 1), (0, 0, 0), 4, False),
    # groups=4, with padding and bias
    (1, 8, 16, 4, 6, 6, 3, 3, 3, (1, 1, 1), (1, 1, 1), 4, True),
    # depthwise: groups=C_in=C_out
    (1, 4, 4, 4, 5, 5, 2, 3, 3, (1, 1, 1), (0, 0, 0), 4, False),
    # depthwise with padding and bias
    (1, 8, 8, 4, 6, 6, 3, 3, 3, (1, 1, 1), (1, 1, 1), 8, True),
    # depthwise with stride
    (1, 4, 4, 6, 6, 6, 3, 3, 3, (2, 2, 2), (1, 1, 1), 4, False),
    # larger channel counts, groups=2
    (1, 16, 32, 4, 6, 6, 3, 3, 3, (1, 1, 1), (1, 1, 1), 2, False),
    # 1x1x1 pointwise with groups
    (1, 8, 16, 4, 5, 5, 1, 1, 1, (1, 1, 1), (0, 0, 0), 4, False),
    # temporal-only kernel with groups
    (1, 8, 8, 6, 4, 4, 3, 1, 1, (1, 1, 1), (1, 0, 0), 4, False),
]

GROUPED_PARAM_NAMES = "B, C_in, C_out, D, H, W, kD, kH, kW, stride, padding, groups, use_bias"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestConv3dGrouped:
    """Test grouped convolution support against PyTorch F.conv3d."""

    @pytest.mark.parametrize(GROUPED_PARAM_NAMES, GROUPED_CONV_PARAMS)
    def test_ref_grouped_vs_pytorch(self, B, C_in, C_out, D, H, W, kD, kH, kW,
                                     stride, padding, groups, use_bias):
        """Validate NumPy reference with groups against PyTorch."""
        input_np, weight_np, bias_np = _generate_grouped_inputs(
            B, C_in, C_out, D, H, W, kD, kH, kW, groups, use_bias
        )

        out_ref = conv3d_ref(input_np, weight_np, bias_np,
                             stride=stride, padding=padding, groups=groups)

        input_t = torch.from_numpy(input_np)
        weight_t = torch.from_numpy(weight_np)
        bias_t = torch.from_numpy(bias_np) if bias_np is not None else None
        out_torch = F.conv3d(input_t, weight_t, bias_t,
                             stride=stride, padding=padding, groups=groups).numpy()

        np.testing.assert_allclose(out_ref, out_torch, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(GROUPED_PARAM_NAMES, GROUPED_CONV_PARAMS)
    @pytest.mark.skipif(not HAS_NKI, reason="NKI not installed")
    def test_nki_grouped_vs_pytorch(self, B, C_in, C_out, D, H, W, kD, kH, kW,
                                     stride, padding, groups, use_bias):
        """Validate NKI conv3d with groups against PyTorch."""
        input_np, weight_np, bias_np = _generate_grouped_inputs(
            B, C_in, C_out, D, H, W, kD, kH, kW, groups, use_bias
        )

        input_t = torch.from_numpy(input_np)
        weight_t = torch.from_numpy(weight_np)
        bias_t = torch.from_numpy(bias_np) if bias_np is not None else None
        expected = F.conv3d(input_t, weight_t, bias_t,
                            stride=stride, padding=padding, groups=groups).numpy()

        actual = conv3d_nki(input_np, weight_np, bias_np,
                            stride=stride, padding=padding, groups=groups)

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    def test_groups_1_matches_standard(self):
        """groups=1 should produce identical results to the default (no groups arg)."""
        B, C_in, C_out = 1, 3, 4
        D, H, W = 4, 6, 6
        kD, kH, kW = 2, 3, 3

        input_np, weight_np, bias_np = _generate_inputs(
            B, C_in, C_out, D, H, W, kD, kH, kW, use_bias=True
        )

        out_default = conv3d_ref(input_np, weight_np, bias_np,
                                 stride=(1, 1, 1), padding=(0, 0, 0))
        out_groups1 = conv3d_ref(input_np, weight_np, bias_np,
                                 stride=(1, 1, 1), padding=(0, 0, 0), groups=1)

        np.testing.assert_allclose(out_default, out_groups1, rtol=1e-7, atol=1e-7)

    def test_grouped_output_shape(self):
        """Verify output shape is the same regardless of groups."""
        B, C_in, C_out = 1, 8, 16
        D, H, W = 4, 6, 6
        kD, kH, kW = 3, 3, 3
        groups = 4

        input_np, weight_np, _ = _generate_grouped_inputs(
            B, C_in, C_out, D, H, W, kD, kH, kW, groups, use_bias=False
        )

        out = conv3d_ref(input_np, weight_np,
                         stride=(1, 1, 1), padding=(1, 1, 1), groups=groups)

        assert out.shape == (B, C_out, D, H, W), (
            f"Expected shape ({B},{C_out},{D},{H},{W}), got {out.shape}"
        )

    def test_depthwise_each_channel_independent(self):
        """In depthwise conv, zeroing one input channel should not affect others."""
        B, C, D, H, W = 1, 4, 4, 5, 5
        kD, kH, kW = 2, 3, 3

        rng = np.random.RandomState(42)
        input_np = rng.randn(B, C, D, H, W).astype(np.float32)
        # Depthwise: weight shape [C, 1, kD, kH, kW]
        weight_np = rng.randn(C, 1, kD, kH, kW).astype(np.float32)

        # Full output
        out_full = conv3d_ref(input_np, weight_np, groups=C)

        # Zero out channel 0 input
        input_zeroed = input_np.copy()
        input_zeroed[:, 0] = 0.0
        out_zeroed = conv3d_ref(input_zeroed, weight_np, groups=C)

        # Channel 0 output should change, channels 1-3 should be identical
        assert not np.allclose(out_full[:, 0], out_zeroed[:, 0]), (
            "Channel 0 output should change when its input is zeroed"
        )
        np.testing.assert_allclose(
            out_full[:, 1:], out_zeroed[:, 1:], rtol=1e-7, atol=1e-7
        )
