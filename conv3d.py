# Copyright 2026 Lianyu Huang
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

Algorithm (v2 - host im2col):
    The conv3d() wrapper constructs im2col matrices on the host (NumPy),
    then calls a simple tiled matmul NKI kernel with fixed tile sizes.
    This avoids variable-bound nl.affine_range inside the kernel, which
    causes OOB errors when tile dimensions exceed 128.

    For each batch element and output depth position:
        1. Build im2col columns across all kD temporal kernel positions
        2. Concatenate weight slices across kD
        3. Pad K (contraction) and M (C_out) to multiples of 128
        4. Call tiled_matmul_kernel: output = w_padded.T @ im_padded
        5. Extract valid region, add bias, store result

Layouts:
    Input:  [B, C_in, D, H, W]            (NCDHW, already padded if needed)
    Weight: [C_out, C_in, kD, kH, kW]
    Bias:   [C_out]
    Output: [B, C_out, D_out, H_out, W_out]

Supports: arbitrary kernel sizes, strides, symmetric padding, dilation, bias, f32/bf16,
grouped convolution (including depthwise).
"""

import math

import numpy as np

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl


def _div_ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


def _pad_to_multiple(x: int, m: int) -> int:
    """Round x up to the next multiple of m."""
    return _div_ceil(x, m) * m


def _build_im2col_2d_numpy(
    input_frame, kH, kW, stride_h, stride_w, H_out, W_out,
    dilation_h=1, dilation_w=1,
):
    """Build im2col matrix for a single 2D spatial frame (no padding needed, input already padded).

    Args:
        input_frame: (C_in, H, W) numpy array (already padded spatially)
        kH, kW: spatial kernel dims
        stride_h, stride_w: spatial strides
        H_out, W_out: output spatial dims
        dilation_h, dilation_w: spatial dilation rates (default 1)

    Returns:
        col: (C_in * kH * kW, H_out * W_out) numpy array
    """
    C_in, H, W = input_frame.shape
    col_height = C_in * kH * kW
    spatial_out = H_out * W_out

    col = np.zeros((col_height, spatial_out), dtype=input_frame.dtype)
    for c in range(C_in):
        for kh in range(kH):
            for kw in range(kW):
                row = c * kH * kW + kh * kW + kw
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_in = h_out * stride_h + kh * dilation_h
                        w_in = w_out * stride_w + kw * dilation_w
                        sp_idx = h_out * W_out + w_out
                        col[row, sp_idx] = input_frame[c, h_in, w_in]
    return col


# ---------------------------------------------------------------------------
# Tiled matmul NKI kernel (v2) - fixed tile sizes, no variable bounds
# ---------------------------------------------------------------------------

@nki.jit
def tiled_matmul_kernel(w_ref, im_ref):
    """Tiled matrix multiplication kernel: output = w_ref.T @ im_ref.

    Both K and M dimensions must be multiples of 128 (padded on host).
    N (free dimension) must be <= 512.

    Args:
        w_ref:  [K_padded, M_padded] stationary matrix (weights) on HBM
        im_ref: [K_padded, N] moving matrix (im2col columns) on HBM

    Returns:
        output: [M_padded, N] result matrix on HBM
    """
    K = w_ref.shape[0]
    M = w_ref.shape[1]
    N = im_ref.shape[1]

    P_MAX = nl.tile_size.pmax  # 128

    n_k_tiles = K // P_MAX
    n_m_tiles = M // P_MAX

    output = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.hbm)

    # Index patterns for bulk load/store
    i_p = nl.arange(P_MAX)[:, None]
    i_f = nl.arange(N)[None, :]

    for m_idx in range(n_m_tiles):
        # Accumulator in PSUM
        acc = nl.zeros((P_MAX, N), dtype=nl.float32, buffer=nl.psum)

        for k_idx in range(n_k_tiles):
            # Load weight tile: [P_MAX, P_MAX] from w_ref[k_block, m_block]
            w_tile = nl.load(
                w_ref[k_idx * P_MAX + i_p, m_idx * P_MAX + nl.arange(P_MAX)[None, :]]
            )
            # Load im2col tile: [P_MAX, N] from im_ref[k_block, :]
            im_tile = nl.load(im_ref[k_idx * P_MAX + i_p, i_f])
            # nc_matmul: stationary.T @ moving -> [P_MAX(M), N]
            acc += nisa.nc_matmul(w_tile, im_tile)

        # Copy from PSUM to SBUF then store to HBM
        result = nisa.tensor_copy(acc)
        nl.store(output[m_idx * P_MAX + i_p, i_f], value=result)

    return output


# ---------------------------------------------------------------------------
# Legacy kernel (v1) - kept for backward compatibility / import
# ---------------------------------------------------------------------------

@nki.jit
def conv3d_kernel(
    input_ref,
    weight_ref,
    bias_ref=None,
    stride_d=1,
    stride_h=1,
    stride_w=1,
):
    """
    3D convolution kernel (v1, element-wise im2col inside kernel).

    Note: This kernel fails for large C_in/C_out due to variable-bound
    nl.affine_range. Kept for backward compatibility (test imports it).
    The conv3d() wrapper now uses tiled_matmul_kernel instead.
    """
    B = input_ref.shape[0]
    C_in = input_ref.shape[1]
    D = input_ref.shape[2]
    H = input_ref.shape[3]
    W = input_ref.shape[4]
    C_out = weight_ref.shape[0]
    kD = weight_ref.shape[2]
    kH = weight_ref.shape[3]
    kW = weight_ref.shape[4]

    D_out = (D - kD) // stride_d + 1
    H_out = (H - kH) // stride_h + 1
    W_out = (W - kW) // stride_w + 1

    P_MAX = nl.tile_size.pmax
    F_MAX = nl.tile_size.psum_fmax

    col_height = C_in * kH * kW
    spatial_out = H_out * W_out

    c_out_tile = min(C_out, P_MAX)
    col_tile = min(col_height, P_MAX)
    sp_tile = min(spatial_out, F_MAX)

    n_co_tiles = _div_ceil(C_out, c_out_tile)
    n_col_tiles = _div_ceil(col_height, col_tile)
    n_sp_tiles = _div_ceil(spatial_out, sp_tile)

    output = nl.ndarray(
        (B, C_out, D_out, H_out, W_out), dtype=input_ref.dtype, buffer=nl.hbm
    )

    for b in range(B):
        for d_out in range(D_out):
            for co_idx in range(n_co_tiles):
                co_start = co_idx * c_out_tile
                co_size = min(c_out_tile, C_out - co_start)
                for sp_idx in range(n_sp_tiles):
                    sp_start = sp_idx * sp_tile
                    sp_size = min(sp_tile, spatial_out - sp_start)
                    acc = nl.zeros(
                        (c_out_tile, sp_tile), dtype=nl.float32, buffer=nl.psum
                    )
                    for kd in range(kD):
                        d_in = d_out * stride_d + kd
                        for col_idx in range(n_col_tiles):
                            col_start = col_idx * col_tile
                            col_size = min(col_tile, col_height - col_start)
                            im2col = nl.zeros(
                                (col_tile, sp_tile), dtype=input_ref.dtype,
                                buffer=nl.sbuf
                            )
                            for local_col in nl.affine_range(col_size):
                                flat_idx = col_start + local_col
                                c = flat_idx // (kH * kW)
                                kh = (flat_idx % (kH * kW)) // kW
                                kw = flat_idx % kW
                                for local_sp in nl.affine_range(sp_size):
                                    sp_flat = sp_start + local_sp
                                    h_out = sp_flat // W_out
                                    w_out = sp_flat % W_out
                                    h_in = h_out * stride_h + kh
                                    w_in = w_out * stride_w + kw
                                    im2col[local_col, local_sp] = nl.load(
                                        input_ref[b, c, d_in, h_in, w_in]
                                    )
                            w_tile = nl.zeros(
                                (col_tile, c_out_tile), dtype=weight_ref.dtype,
                                buffer=nl.sbuf
                            )
                            for local_co in nl.affine_range(co_size):
                                for local_col in nl.affine_range(col_size):
                                    flat_idx = col_start + local_col
                                    c = flat_idx // (kH * kW)
                                    kh = (flat_idx % (kH * kW)) // kW
                                    kw = flat_idx % kW
                                    w_tile[local_col, local_co] = nl.load(
                                        weight_ref[co_start + local_co, c, kd, kh, kw]
                                    )
                            acc += nisa.nc_matmul(
                                w_tile[0:col_size, 0:co_size],
                                im2col[0:col_size, 0:sp_size],
                            )
                    if bias_ref is not None:
                        bias_tile = nl.ndarray(
                            (c_out_tile, 1), dtype=nl.float32, buffer=nl.sbuf
                        )
                        for local_co in nl.affine_range(co_size):
                            bias_tile[local_co, 0] = nl.load(
                                bias_ref[co_start + local_co]
                            )
                        for local_co in nl.affine_range(co_size):
                            for local_sp in nl.affine_range(sp_size):
                                acc[local_co, local_sp] = (
                                    acc[local_co, local_sp] + bias_tile[local_co, 0]
                                )
                    result = nisa.tensor_copy(acc[0:co_size, 0:sp_size])
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


# ---------------------------------------------------------------------------
# User-facing API (v2 - host im2col + tiled matmul kernel)
# ---------------------------------------------------------------------------

def conv3d(input_np, weight_np, bias_np=None,
           stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1),
           groups=1):
    """
    Conv3d with padding, dilation, and grouped convolution support.

    Performs im2col on the host (NumPy), pads to multiples of 128,
    then calls tiled_matmul_kernel for the GEMM. This avoids
    variable-bound loops inside the NKI kernel.

    For grouped convolution (groups > 1), the input and output channels
    are split into `groups` independent groups, each processed separately.
    Depthwise convolution is the special case where groups == C_in == C_out.

    Args:
        input_np:  [B, C_in, D, H, W] numpy array
        weight_np: [C_out, C_in/groups, kD, kH, kW] numpy array
        bias_np:   [C_out] numpy array or None
        stride:    (stride_d, stride_h, stride_w)
        padding:   (pad_d, pad_h, pad_w)
        dilation:  (dilation_d, dilation_h, dilation_w)
        groups:    number of groups for grouped convolution (default: 1)

    Returns:
        output: [B, C_out, D_out, H_out, W_out] numpy array
    """
    pd, ph, pw = padding
    sd, sh, sw = stride
    dd, dh, dw = dilation

    B, C_in, D, H, W = input_np.shape
    C_out = weight_np.shape[0]
    C_in_per_group = weight_np.shape[1]
    kD, kH, kW = weight_np.shape[2], weight_np.shape[3], weight_np.shape[4]

    assert C_in % groups == 0, (
        f"C_in ({C_in}) must be divisible by groups ({groups})"
    )
    assert C_out % groups == 0, (
        f"C_out ({C_out}) must be divisible by groups ({groups})"
    )
    assert C_in_per_group == C_in // groups, (
        f"Weight C_in dim ({C_in_per_group}) must equal C_in/groups ({C_in // groups})"
    )

    C_out_per_group = C_out // groups

    # Pad input if needed
    if pd > 0 or ph > 0 or pw > 0:
        input_padded = np.pad(
            input_np,
            ((0, 0), (0, 0), (pd, pd), (ph, ph), (pw, pw)),
            mode="constant",
        )
    else:
        input_padded = input_np

    _, _, D_pad, H_pad, W_pad = input_padded.shape

    # Effective kernel size accounts for dilation
    kD_eff = kD + (kD - 1) * (dd - 1)
    kH_eff = kH + (kH - 1) * (dh - 1)
    kW_eff = kW + (kW - 1) * (dw - 1)

    D_out = (D_pad - kD_eff) // sd + 1
    H_out = (H_pad - kH_eff) // sh + 1
    W_out = (W_pad - kW_eff) // sw + 1
    spatial_out = H_out * W_out

    # Allocate output
    output = np.zeros((B, C_out, D_out, H_out, W_out), dtype=np.float32)

    # Process each group independently
    for g in range(groups):
        cin_start = g * C_in_per_group
        cin_end = cin_start + C_in_per_group
        cout_start = g * C_out_per_group
        cout_end = cout_start + C_out_per_group

        # Slice weight for this group: [C_out_per_group, C_in_per_group, kD, kH, kW]
        w_group = weight_np[cout_start:cout_end]

        # Concatenate weight across kD
        w_cat = w_group.transpose(0, 2, 1, 3, 4).reshape(
            C_out_per_group, kD * C_in_per_group * kH * kW
        )

        K_total = kD * C_in_per_group * kH * kW

        # Transpose for nc_matmul: stationary input is [K, M] where M = C_out_per_group
        w_for_matmul = w_cat.T  # [K_total, C_out_per_group]

        # Pad K and M to multiples of 128
        P = 128
        K_padded = _pad_to_multiple(K_total, P)
        M_padded = _pad_to_multiple(C_out_per_group, P)

        # Pad weight matrix
        w_padded = np.zeros((K_padded, M_padded), dtype=weight_np.dtype)
        w_padded[:K_total, :C_out_per_group] = w_for_matmul

        # Slice bias for this group
        bias_group = None
        if bias_np is not None:
            bias_group = bias_np[cout_start:cout_end]

        for b in range(B):
            for d_out in range(D_out):
                # Build concatenated im2col across all kD positions (group channels only)
                im_cols = []
                for kd in range(kD):
                    d_in = d_out * sd + kd * dd
                    # Only use channels for this group
                    frame = input_padded[b, cin_start:cin_end, d_in, :, :]
                    col = _build_im2col_2d_numpy(
                        frame, kH, kW, sh, sw, H_out, W_out,
                        dilation_h=dh, dilation_w=dw,
                    )  # [C_in_per_group*kH*kW, spatial_out]
                    im_cols.append(col)

                im_cat = np.concatenate(im_cols, axis=0)  # [K_total, spatial_out]

                # Pad im2col matrix: K to K_padded (N stays as-is)
                im_padded = np.zeros((K_padded, spatial_out), dtype=input_np.dtype)
                im_padded[:K_total, :] = im_cat

                # Call tiled matmul kernel
                result = nki.simulate_kernel(
                    tiled_matmul_kernel, w_padded, im_padded
                )  # [M_padded, spatial_out]

                # Extract valid region: [C_out_per_group, spatial_out]
                out_slice = result[:C_out_per_group, :spatial_out]

                # Add bias
                if bias_group is not None:
                    out_slice = out_slice + bias_group[:, np.newaxis]

                # Reshape and store
                output[b, cout_start:cout_end, d_out, :, :] = out_slice.reshape(
                    C_out_per_group, H_out, W_out
                )

    return output
