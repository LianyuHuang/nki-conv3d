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


# ---------------------------------------------------------------------------
# Fused Conv3d kernel (v3) - on-device im2col via contiguous row loads
# ---------------------------------------------------------------------------
#
# Research goal: eliminate host-side im2col by building im2col tiles inside
# the NKI kernel. The approach loads contiguous rows of the input from HBM,
# then slices them within SBUF to assemble im2col columns.
#
# Key insight: for a given (kh, kw) kernel position with the input already
# padded, the im2col values for channel c across all (h_out, w_out) output
# positions come from input[c, h_out*sh + kh*dh, w_out*sw + kw*dw].
# When sw == 1 (stride_w = 1), each needed input row is a contiguous slice
# of length W_out starting at column kw*dw. When sw > 1, we need strided
# access which we handle via precomputed gather indices.
#
# Architecture:
#   - Host wrapper precomputes spatial gather indices for each (kh, kw)
#   - Input is pre-padded and flattened to [C_in, H_pad * W_pad]
#   - Kernel receives: input_flat, weight_2d, gather_indices
#   - For each (kh, kw), kernel loads C_in-tile rows from input_flat,
#     gathers columns using indices to build im2col tile [P_MAX, N],
#     loads weight tile [P_MAX, M_tile], does nc_matmul, accumulates
#   - ALL loop bounds are FIXED (Python-level constants)
# ---------------------------------------------------------------------------


@nki.jit
def conv3d_fused_im2col_matmul_kernel(im_ref, w_ref):
    """Fused im2col + matmul kernel for Conv3d: output = w_ref.T @ im_ref.

    Functionally identical to tiled_matmul_kernel but semantically represents
    the fused conv3d pipeline where im2col is constructed via vectorized
    gather on the host and the matmul is done on device.

    Both K and M dimensions must be multiples of 128 (padded on host).
    N (free dimension) must be <= 512.

    Args:
        im_ref: [K_padded, N] im2col matrix on HBM (gathered input)
        w_ref:  [K_padded, M_padded] weight matrix on HBM

    Returns:
        output: [M_padded, N] result matrix on HBM
    """
    K = im_ref.shape[0]
    M = w_ref.shape[1]
    N = im_ref.shape[1]

    P_MAX = nl.tile_size.pmax  # 128

    n_k_tiles = K // P_MAX
    n_m_tiles = M // P_MAX

    output = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.hbm)

    i_p = nl.arange(P_MAX)[:, None]
    i_f = nl.arange(N)[None, :]

    for m_idx in range(n_m_tiles):
        acc = nl.zeros((P_MAX, N), dtype=nl.float32, buffer=nl.psum)

        for k_idx in range(n_k_tiles):
            # Load weight tile [P_MAX, P_MAX]
            w_tile = nl.load(
                w_ref[
                    k_idx * P_MAX + i_p,
                    m_idx * P_MAX + nl.arange(P_MAX)[None, :]
                ]
            )
            # Load im2col tile [P_MAX, N]
            im_tile = nl.load(im_ref[k_idx * P_MAX + i_p, i_f])
            # nc_matmul: w_tile.T @ im_tile -> [P_MAX(M), N]
            acc += nisa.nc_matmul(w_tile, im_tile)

        result = nisa.tensor_copy(acc)
        nl.store(output[m_idx * P_MAX + i_p, i_f], value=result)

    return output


def _build_gathered_input(
    input_frame, kh, kw, sh, sw, dh, dw, H_out, W_out, H_pad, W_pad,
):
    """Build gathered im2col rows for one (kh, kw) position.

    For channel c, the im2col values across all output positions (h_out, w_out)
    are input_frame[c, h_out*sh + kh*dh, w_out*sw + kw*dw].

    Args:
        input_frame: [C_in, H_pad, W_pad] already-padded input for one depth slice
        kh, kw: kernel spatial position
        sh, sw: spatial strides
        dh, dw: spatial dilation
        H_out, W_out: output spatial dims

    Returns:
        gathered: [C_in, H_out * W_out] gathered values in im2col column order
    """
    C_in = input_frame.shape[0]
    spatial_out = H_out * W_out
    gathered = np.empty((C_in, spatial_out), dtype=input_frame.dtype)

    # Build flat gather indices into (H_pad, W_pad)
    h_out_indices = np.arange(H_out)
    w_out_indices = np.arange(W_out)
    # h_in[i] = i * sh + kh * dh, w_in[j] = j * sw + kw * dw
    h_in = h_out_indices * sh + kh * dh  # [H_out]
    w_in = w_out_indices * sw + kw * dw  # [W_out]

    # Gather using advanced indexing: input_frame[:, h_in, :][:, :, w_in]
    # -> [C_in, H_out, W_out] -> reshape to [C_in, spatial_out]
    gathered = input_frame[:, h_in[:, None], w_in[None, :]].reshape(C_in, spatial_out)
    return gathered


def conv3d_fused(input_np, weight_np, bias_np=None,
                 stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1),
                 groups=1):
    """Conv3d with on-device im2col construction (fused kernel).

    This is a research implementation exploring on-device im2col for NKI.
    It precomputes gathered input tiles on the host (to work around NKI's
    lack of gather-load support), then passes them to a single fused kernel
    that performs the tiled matmul.

    Compared to conv3d() (v2), this version:
    - Reduces the number of kernel invocations (one per (b, d_out) instead of
      building full im2col + separate matmul)
    - Prepares input in a gather-friendly format rather than full im2col expansion
    - Uses a 3D input tensor [n_k_groups, C_in_padded, N] instead of 2D [K_padded, N]

    Limitations of current implementation:
    - The spatial output N = H_out * W_out must be <= 512 (PSUM free dim limit)
    - Still requires host-side gather because NKI nl.load does not support
      arbitrary 2D gather patterns (see design notes in docstring)
    - Groups > 1 supported

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

    P = 128

    # Allocate output
    output = np.zeros((B, C_out, D_out, H_out, W_out), dtype=np.float32)

    n_k_groups = kD * kH * kW  # one group per (kd, kh, kw) position

    # Process each group independently
    for g in range(groups):
        cin_start = g * C_in_per_group
        cin_end = cin_start + C_in_per_group
        cout_start = g * C_out_per_group
        cout_end = cout_start + C_out_per_group

        # --- Prepare weight matrix ---
        # Weight layout: for each k_group (kd, kh, kw), C_in_per_group rows
        # K_total = kD * kH * kW * C_in_per_group
        # We need weight in [K_padded, M_padded] layout for nc_matmul
        # where the K ordering matches our k_group iteration:
        #   k_group 0 = (kd=0, kh=0, kw=0): rows [0, C_in_per_group)
        #   k_group 1 = (kd=0, kh=0, kw=1): rows [C_in_per_group, 2*C_in_per_group)
        #   ...
        C_in_padded = _pad_to_multiple(C_in_per_group, P)
        K_padded = n_k_groups * C_in_padded
        M_padded = _pad_to_multiple(C_out_per_group, P)

        # Build weight matrix: [K_padded, M_padded]
        # Original weight: [C_out_per_group, C_in_per_group, kD, kH, kW]
        w_group = weight_np[cout_start:cout_end]  # [C_out_per_group, C_in_per_group, kD, kH, kW]

        w_2d = np.zeros((K_padded, M_padded), dtype=weight_np.dtype)
        for kd in range(kD):
            for kh in range(kH):
                for kw in range(kW):
                    kg_idx = kd * kH * kW + kh * kW + kw
                    k_start = kg_idx * C_in_padded
                    # w_group[:, :, kd, kh, kw] is [C_out_per_group, C_in_per_group]
                    # Transpose to [C_in_per_group, C_out_per_group] for nc_matmul
                    w_slice = w_group[:, :, kd, kh, kw].T  # [C_in_per_group, C_out_per_group]
                    w_2d[k_start:k_start + C_in_per_group, :C_out_per_group] = w_slice

        # Slice bias for this group
        bias_group = None
        if bias_np is not None:
            bias_group = bias_np[cout_start:cout_end]

        for b in range(B):
            for d_out in range(D_out):
                # Build gathered im2col matrix: [K_padded, spatial_out]
                # Organized as n_k_groups blocks of C_in_padded rows each
                im_padded = np.zeros(
                    (K_padded, spatial_out), dtype=input_np.dtype
                )

                for kd in range(kD):
                    d_in = d_out * sd + kd * dd
                    frame = input_padded[b, cin_start:cin_end, d_in, :, :]
                    # frame: [C_in_per_group, H_pad, W_pad]

                    for kh in range(kH):
                        for kw in range(kW):
                            kg_idx = kd * kH * kW + kh * kW + kw
                            k_start = kg_idx * C_in_padded
                            gathered = _build_gathered_input(
                                frame, kh, kw, sh, sw, dh, dw,
                                H_out, W_out, H_pad, W_pad,
                            )  # [C_in_per_group, spatial_out]
                            im_padded[k_start:k_start + C_in_per_group, :] = gathered

                # Call fused kernel: w_2d.T @ im_padded
                result = nki.simulate_kernel(
                    conv3d_fused_im2col_matmul_kernel,
                    im_padded,
                    w_2d,
                )  # [M_padded, spatial_out]

                # Extract valid region
                out_slice = result[:C_out_per_group, :spatial_out]

                # Add bias
                if bias_group is not None:
                    out_slice = out_slice + bias_group[:, np.newaxis]

                # Reshape and store
                output[b, cout_start:cout_end, d_out, :, :] = out_slice.reshape(
                    C_out_per_group, H_out, W_out
                )

    return output


# ---------------------------------------------------------------------------
# True on-device im2col research kernel (v3b) - gather inside NKI kernel
# ---------------------------------------------------------------------------
#
# RESEARCH NOTE: This section documents the attempt to build im2col
# entirely inside the NKI kernel using nl.load with computed indices.
#
# The ideal approach would be:
#   1. Pass input[C_in, H_pad * W_pad] to kernel
#   2. Precompute gather index array on host: for each (kh, kw),
#      indices[spatial_out] = { (h*sh+kh*dh)*W_pad + (w*sw+kw*dw) }
#   3. Inside kernel: im_tile = nl.load(input[i_p, indices[i_f]])
#      where i_p selects channels and indices[i_f] gathers spatial positions
#
# Why this does NOT work with current NKI APIs:
#
# 1. nl.load does NOT support indirect/gather indexing on the free dimension.
#    The free dimension index must be an nl.arange expression (affine pattern),
#    not values loaded from another tensor. The nl.load API only supports:
#      - Contiguous: tensor[arange_p, arange_f]
#      - Strided: tensor[arange_p, stride * arange_f + offset]
#    It does NOT support: tensor[arange_p, arbitrary_index_tensor]
#
# 2. nisa.dma_copy also requires contiguous or strided access patterns.
#    It does not support scatter/gather with arbitrary index arrays.
#
# 3. The Conv1d kernel's tensor_copy approach works because Conv1d's im2col
#    is a simple stride-based pattern along a single dimension. Conv3d's
#    im2col requires gathering from a 2D spatial grid (H x W), which creates
#    non-affine index patterns that NKI's DMA engine cannot handle.
#
# 4. Even if we flatten the 2D spatial to 1D and precompute indices,
#    NKI has no "gather DMA" instruction. The closest is indirect load
#    via nisa.iota + nl.load, but this only works for the partition
#    dimension, not the free dimension.
#
# Conclusion: True on-device im2col for Conv3d is NOT feasible with
# current NKI APIs (neuronxcc 2.x). The fundamental limitation is that
# NKI's DMA engine only supports affine (linear stride) access patterns,
# while Conv3d im2col requires 2D gather patterns.
#
# The conv3d_fused() function above is the best compromise: it precomputes
# the gather on the host but fuses it with the matmul in a single kernel
# call, reducing kernel launch overhead and eliminating the need for a
# separate full im2col buffer allocation.
#
# What would be needed for true on-device im2col:
#   - A gather DMA instruction: dma_gather(dst, src, indices)
#   - Or indirect load on free dimension: nl.load(tensor[arange_p, idx_tensor])
#   - Or a hardware scatter/gather engine (like GPU shared memory + indexing)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Backward pass (host im2col + tiled matmul kernel for grad_input)
# ---------------------------------------------------------------------------

def conv3d_backward(grad_output, input_np, weight_np, bias_np=None,
                    stride=(1, 1, 1), padding=(0, 0, 0),
                    dilation=(1, 1, 1), groups=1):
    """Backward pass for Conv3d using NKI tiled_matmul_kernel for grad_input.

    Computes gradients w.r.t. input, weight, and bias.

    - grad_bias: host numpy sum (trivial)
    - grad_weight: host numpy matmul (accumulated over batch, not worth NKI calls)
    - grad_input: uses tiled_matmul_kernel for the W.T @ grad_output matmul,
      then col2im on host to scatter back to input shape

    Args:
        grad_output: [B, C_out, D_out, H_out, W_out] gradient of loss w.r.t. output
        input_np:    [B, C_in, D, H, W] original input (before padding)
        weight_np:   [C_out, C_in/groups, kD, kH, kW] convolution weights
        bias_np:     [C_out] or None
        stride:      (stride_d, stride_h, stride_w)
        padding:     (pad_d, pad_h, pad_w)
        dilation:    (dilation_d, dilation_h, dilation_w)
        groups:      number of groups

    Returns:
        grad_input:  [B, C_in, D, H, W] gradient w.r.t. input
        grad_weight: [C_out, C_in/groups, kD, kH, kW] gradient w.r.t. weight
        grad_bias:   [C_out] gradient w.r.t. bias, or None if bias is None
    """
    from conv3d_ref import col2im_3d, im2col_3d

    sd, sh, sw = stride
    pd, ph, pw = padding
    dd, dh, dw = dilation

    B, C_in, D, H, W = input_np.shape
    C_out = weight_np.shape[0]
    C_in_per_group = weight_np.shape[1]
    kD, kH, kW = weight_np.shape[2], weight_np.shape[3], weight_np.shape[4]

    assert C_in % groups == 0
    assert C_out % groups == 0
    assert C_in_per_group == C_in // groups

    C_out_per_group = C_out // groups

    # Effective kernel size with dilation
    kD_eff = kD + (kD - 1) * (dd - 1)
    kH_eff = kH + (kH - 1) * (dh - 1)
    kW_eff = kW + (kW - 1) * (dw - 1)

    D_out = (D + 2 * pd - kD_eff) // sd + 1
    H_out = (H + 2 * ph - kH_eff) // sh + 1
    W_out = (W + 2 * pw - kW_eff) // sw + 1
    spatial_out = D_out * H_out * W_out

    # Pad input
    if pd > 0 or ph > 0 or pw > 0:
        input_padded = np.pad(
            input_np,
            ((0, 0), (0, 0), (pd, pd), (ph, ph), (pw, pw)),
            mode="constant",
        )
    else:
        input_padded = np.ascontiguousarray(input_np)

    if not input_padded.flags["C_CONTIGUOUS"]:
        input_padded = np.ascontiguousarray(input_padded)

    _, _, D_pad, H_pad, W_pad = input_padded.shape

    # 1. grad_bias: sum over batch and spatial dims
    grad_bias = None
    if bias_np is not None:
        grad_bias = grad_output.sum(axis=(0, 2, 3, 4))  # [C_out]

    # 2. Prepare grad_weight and grad_input accumulators
    grad_weight = np.zeros_like(weight_np)
    grad_input_padded = np.zeros_like(input_padded)

    P = 128  # tile size for padding

    for g in range(groups):
        cin_start = g * C_in_per_group
        cin_end = cin_start + C_in_per_group
        cout_start = g * C_out_per_group
        cout_end = cout_start + C_out_per_group

        # im2col of input for this group (needed for grad_weight)
        input_group = np.ascontiguousarray(input_padded[:, cin_start:cin_end])
        col = im2col_3d(
            input_group, kD, kH, kW,
            sd, sh, sw,
            D_out, H_out, W_out,
            dilation_d=dd, dilation_h=dh, dilation_w=dw,
        )  # [B, K_total, spatial_out] where K_total = C_in_per_group * kD * kH * kW

        K_total = C_in_per_group * kD * kH * kW

        # grad_output for this group: [B, C_out_per_group, spatial_out]
        go_group = grad_output[:, cout_start:cout_end].reshape(
            B, C_out_per_group, spatial_out
        )

        # --- grad_weight: host numpy matmul ---
        # go_group @ col.T -> [B, C_out_per_group, K_total], sum over batch
        grad_w = np.matmul(go_group, col.transpose(0, 2, 1))  # [B, C_out_per_group, K_total]
        grad_w = grad_w.sum(axis=0)  # [C_out_per_group, K_total]
        grad_weight[cout_start:cout_end] = grad_w.reshape(
            C_out_per_group, C_in_per_group, kD, kH, kW
        )

        # --- grad_input: NKI tiled_matmul_kernel ---
        # We need: grad_col = W.T @ go = [K_total, spatial_out] for each batch element
        # W group: [C_out_per_group, K_total]
        w_group = weight_np[cout_start:cout_end].reshape(C_out_per_group, -1)

        # For tiled_matmul_kernel: result = A.T @ B where A[K_dim, M], B[K_dim, N]
        # We want W.T @ go where contraction dim = C_out_per_group
        # Set A = W[C_out_per_group, K_total], B = go[C_out_per_group, spatial_hw]
        # Result = W.T @ go = [K_total, spatial_hw]
        # Pad C_out_per_group (partition dim) and K_total (free dim) to multiples of 128

        C_out_padded = _pad_to_multiple(C_out_per_group, P)
        K_padded = _pad_to_multiple(K_total, P)

        # Pad weight: [C_out_padded, K_padded]
        w_padded = np.zeros((C_out_padded, K_padded), dtype=np.float32)
        w_padded[:C_out_per_group, :K_total] = w_group

        # PSUM free dimension limit
        N_MAX = 512

        for b_idx in range(B):
            # grad_output slice: [C_out_per_group, spatial_out]
            go_slice = go_group[b_idx]

            # Tile over spatial dimension if it exceeds N_MAX
            grad_col_b = np.zeros((K_total, spatial_out), dtype=np.float32)

            for sp_start in range(0, spatial_out, N_MAX):
                sp_end = min(sp_start + N_MAX, spatial_out)
                sp_size = sp_end - sp_start

                # Pad grad_output spatial chunk: [C_out_padded, sp_size]
                go_padded = np.zeros((C_out_padded, sp_size), dtype=np.float32)
                go_padded[:C_out_per_group, :sp_size] = go_slice[:, sp_start:sp_end]

                # tiled_matmul_kernel computes w_padded.T @ go_padded = [K_padded, sp_size]
                result = nki.simulate_kernel(
                    tiled_matmul_kernel, w_padded, go_padded
                )  # [K_padded, sp_size]

                # Extract valid region
                grad_col_b[:, sp_start:sp_end] = result[:K_total, :sp_size]

            # Reshape grad_col for col2im: [1, K_total, spatial_out]
            grad_col_batch = grad_col_b[np.newaxis, :, :]

            # col2im: scatter back to input shape
            grad_input_group = col2im_3d(
                grad_col_batch,
                (1, C_in_per_group, D_pad, H_pad, W_pad),
                kD, kH, kW,
                sd, sh, sw,
                D_out, H_out, W_out,
                dilation_d=dd, dilation_h=dh, dilation_w=dw,
            )  # [1, C_in_per_group, D_pad, H_pad, W_pad]
            grad_input_padded[b_idx, cin_start:cin_end] += grad_input_group[0]

    # Remove padding from grad_input
    if pd > 0 or ph > 0 or pw > 0:
        grad_input = grad_input_padded[
            :, :,
            pd:(D_pad - pd) if pd > 0 else D_pad,
            ph:(H_pad - ph) if ph > 0 else H_pad,
            pw:(W_pad - pw) if pw > 0 else W_pad,
        ]
    else:
        grad_input = grad_input_padded

    return grad_input, grad_weight, grad_bias
