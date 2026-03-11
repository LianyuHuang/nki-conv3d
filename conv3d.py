# Copyright 2025 Lianyu Huang
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
3D Convolution NKI kernel for AWS Trainium / NeuronCore.

First NKI Conv3d kernel implementation. Enables video generation models
(Wan2.1/2.2, CogVideoX, HunyuanVideo) to run on Trainium by providing the
missing Conv3d operator required by their 3D VAE components.

Algorithm: Temporal-slice decomposition + spatial im2col + GEMM.
    Conv3d(kD, kH, kW) is exactly decomposed into kD separate spatial
    Conv2d(kH, kW) operations summed across temporal kernel positions.
    Each spatial Conv2d is computed via im2col + nc_matmul.

    output[:, :, d_out, :, :] = sum_{kd=0}^{kD-1}
        Conv2d(input[:, :, d_out*s_d + kd - pad_d, :, :],
               weight[:, :, kd, :, :])

    This decomposition is EXACT (zero numerical error beyond floating point).

Layouts:
    Input:  [B, C_in, D, H, W]            (NCDHW)
    Weight: [C_out, C_in, kD, kH, kW]
    Bias:   [C_out]
    Output: [B, C_out, D_out, H_out, W_out]

Tiling strategy:
    - C_out tiled along partition dimension (max 128)
    - C_in * kH * kW (im2col column height) tiled as contraction dimension
    - H_out * W_out tiled along free dimension (max 512)

Supports: arbitrary kernel sizes, strides, symmetric padding, bias, f32/bf16.
Limitations (v1): no dilation, no grouped convolution.
"""

import math

import numpy as np

import nki
import nki.isa as nisa
import nki.language as nl


def _div_ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


def _build_im2col_2d_numpy(
    input_frame, kH, kW, stride_h, stride_w, pad_h, pad_w, H_out, W_out
):
    """Build im2col matrix for a single 2D spatial frame. (Host-side helper for baremetal.)

    Args:
        input_frame: (C_in, H, W) numpy array
        kH, kW: spatial kernel dims
        stride_h, stride_w: spatial strides
        pad_h, pad_w: spatial padding
        H_out, W_out: output spatial dims

    Returns:
        col: (C_in * kH * kW, H_out * W_out) numpy array
    """
    C_in, H, W = input_frame.shape
    if pad_h > 0 or pad_w > 0:
        input_padded = np.pad(
            input_frame, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant"
        )
    else:
        input_padded = input_frame

    H_pad, W_pad = input_padded.shape[1], input_padded.shape[2]
    col_height = C_in * kH * kW
    spatial_out = H_out * W_out

    col = np.zeros((col_height, spatial_out), dtype=input_frame.dtype)
    for c in range(C_in):
        for kh in range(kH):
            for kw in range(kW):
                row = c * kH * kW + kh * kW + kw
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_in = h_out * stride_h + kh
                        w_in = w_out * stride_w + kw
                        sp_idx = h_out * W_out + w_out
                        col[row, sp_idx] = input_padded[c, h_in, w_in]
    return col


@nki.jit
def conv3d(
    input_ref,
    weight_ref,
    bias_ref=None,
    stride_d=1,
    stride_h=1,
    stride_w=1,
    pad_d=0,
    pad_h=0,
    pad_w=0,
):
    """
    3D convolution via temporal-slice decomposition + im2col + GEMM.

    Args:
        input_ref:  [B, C_in, D, H, W] input tensor on HBM
        weight_ref: [C_out, C_in, kD, kH, kW] filter weights on HBM
        bias_ref:   [C_out] optional bias on HBM (None if no bias)
        stride_d, stride_h, stride_w: strides per dimension
        pad_d, pad_h, pad_w: symmetric padding per dimension

    Returns:
        output: [B, C_out, D_out, H_out, W_out] on HBM
    """
    # --- Dimensions ---
    B = input_ref.shape[0]
    C_in = input_ref.shape[1]
    D = input_ref.shape[2]
    H = input_ref.shape[3]
    W = input_ref.shape[4]
    C_out = weight_ref.shape[0]
    kD = weight_ref.shape[2]
    kH = weight_ref.shape[3]
    kW = weight_ref.shape[4]

    D_out = (D + 2 * pad_d - kD) // stride_d + 1
    H_out = (H + 2 * pad_h - kH) // stride_h + 1
    W_out = (W + 2 * pad_w - kW) // stride_w + 1

    # --- Hardware constants ---
    P_MAX = nl.tile_size.pmax  # 128
    F_MAX = nl.tile_size.psum_fmax  # 512

    # --- Tiling ---
    col_height = C_in * kH * kW
    spatial_out = H_out * W_out

    c_out_tile = min(C_out, P_MAX)
    col_tile = min(col_height, P_MAX)
    sp_tile = min(spatial_out, F_MAX)

    n_co_tiles = _div_ceil(C_out, c_out_tile)
    n_col_tiles = _div_ceil(col_height, col_tile)
    n_sp_tiles = _div_ceil(spatial_out, sp_tile)

    # --- Output allocation ---
    output = nl.ndarray(
        (B, C_out, D_out, H_out, W_out), dtype=input_ref.dtype, buffer=nl.hbm
    )

    # --- Main computation ---
    # For each batch and output depth position, accumulate across kD temporal slices.
    # Each temporal slice contributes a spatial Conv2d computed via im2col + GEMM.

    for b in nl.affine_range(B):
        for d_out in nl.affine_range(D_out):

            # Process C_out in tiles (partition dimension of the GEMM)
            for co_idx in nl.affine_range(n_co_tiles):
                co_start = co_idx * c_out_tile
                co_size = min(c_out_tile, C_out - co_start)

                # Process spatial output in tiles (free dimension of the GEMM)
                for sp_idx in nl.affine_range(n_sp_tiles):
                    sp_start = sp_idx * sp_tile
                    sp_size = min(sp_tile, spatial_out - sp_start)

                    # PSUM accumulator for this output tile (always float32)
                    acc = nl.ndarray(
                        (c_out_tile, sp_tile), dtype=nl.float32, buffer=nl.psum
                    )
                    nisa.memset(acc, 0)

                    # Accumulate over temporal kernel positions
                    for kd in nl.sequential_range(kD):
                        d_in = d_out * stride_d - pad_d + kd

                        # Skip if input depth is out of bounds (zero-padding region)
                        if d_in < 0 or d_in >= D:
                            continue

                        # Accumulate over contraction dimension tiles
                        # (C_in * kH * kW tiled into chunks of col_tile)
                        for col_idx in nl.sequential_range(n_col_tiles):
                            col_start = col_idx * col_tile
                            col_size = min(col_tile, col_height - col_start)

                            # --- Load im2col tile ---
                            # Shape: [col_tile, sp_tile]
                            # Each row corresponds to a (c, kh, kw) position
                            # Each column to a (h_out, w_out) output position
                            im2col = nl.ndarray(
                                (col_tile, sp_tile),
                                dtype=input_ref.dtype,
                                buffer=nl.sbuf,
                            )
                            nisa.memset(im2col, 0)

                            # Fill im2col by loading input values
                            for local_col in nl.affine_range(col_size):
                                flat_idx = col_start + local_col
                                c = flat_idx // (kH * kW)
                                kh = (flat_idx % (kH * kW)) // kW
                                kw = flat_idx % kW

                                for local_sp in nl.affine_range(sp_size):
                                    sp_flat = sp_start + local_sp
                                    h_out = sp_flat // W_out
                                    w_out = sp_flat % W_out

                                    h_in = h_out * stride_h - pad_h + kh
                                    w_in = w_out * stride_w - pad_w + kw

                                    if 0 <= h_in < H and 0 <= w_in < W:
                                        im2col[local_col, local_sp] = nl.load(
                                            input_ref[b, c, d_in, h_in, w_in]
                                        )

                            # --- Load weight tile ---
                            # Shape: [c_out_tile, col_tile]
                            # weight[co, c, kd, kh, kw] mapped to [co, flat_col_idx]
                            w_tile = nl.ndarray(
                                (c_out_tile, col_tile),
                                dtype=weight_ref.dtype,
                                buffer=nl.sbuf,
                            )
                            nisa.memset(w_tile, 0)

                            for local_co in nl.affine_range(co_size):
                                for local_col in nl.affine_range(col_size):
                                    flat_idx = col_start + local_col
                                    c = flat_idx // (kH * kW)
                                    kh = (flat_idx % (kH * kW)) // kW
                                    kw = flat_idx % kW

                                    w_tile[local_co, local_col] = nl.load(
                                        weight_ref[co_start + local_co, c, kd, kh, kw]
                                    )

                            # --- GEMM: w_tile @ im2col -> acc ---
                            # stationary[P, K] @ moving[K, F] -> dst[P, F]
                            # P = c_out, K = col_height, F = spatial_out
                            nisa.nc_matmul(
                                dst=acc[0:co_size, 0:sp_size],
                                stationary=w_tile[0:co_size, 0:col_size],
                                moving=im2col[0:col_size, 0:sp_size],
                            )

                    # --- Add bias ---
                    if bias_ref is not None:
                        bias_tile = nl.ndarray(
                            (c_out_tile, 1), dtype=nl.float32, buffer=nl.sbuf
                        )
                        for local_co in nl.affine_range(co_size):
                            bias_tile[local_co, 0] = nl.load(
                                bias_ref[co_start + local_co]
                            )
                        # Broadcast add: acc[:, :] += bias[:, 0]
                        for local_co in nl.affine_range(co_size):
                            for local_sp in nl.affine_range(sp_size):
                                acc[local_co, local_sp] = (
                                    acc[local_co, local_sp] + bias_tile[local_co, 0]
                                )

                    # --- Store output ---
                    result = nl.ndarray(
                        (c_out_tile, sp_tile),
                        dtype=output.dtype,
                        buffer=nl.sbuf,
                    )
                    nisa.tensor_copy(
                        dst=result[0:co_size, 0:sp_size],
                        src=acc[0:co_size, 0:sp_size],
                    )

                    for local_co in nl.affine_range(co_size):
                        for local_sp in nl.affine_range(sp_size):
                            sp_flat = sp_start + local_sp
                            h_out = sp_flat // W_out
                            w_out = sp_flat % W_out
                            nl.store(
                                output[
                                    b,
                                    co_start + local_co,
                                    d_out,
                                    h_out,
                                    w_out,
                                ],
                                value=result[local_co, local_sp],
                            )

    return output
