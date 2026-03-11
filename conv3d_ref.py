# Copyright 2026 Lianyu Huang
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Pure NumPy reference implementation of Conv3d for testing."""

import numpy as np


def im2col_3d(input_padded, kD, kH, kW, stride_d, stride_h, stride_w, D_out, H_out, W_out):
    """Convert 5D input to column matrix for Conv3d via matmul.

    Args:
        input_padded: (B, C_in, D_pad, H_pad, W_pad) padded input
        kD, kH, kW: kernel dimensions
        stride_d, stride_h, stride_w: strides
        D_out, H_out, W_out: output spatial dimensions

    Returns:
        col: (B, C_in * kD * kH * kW, D_out * H_out * W_out)
    """
    B, C_in, D_pad, H_pad, W_pad = input_padded.shape

    shape = (B, C_in, D_out, H_out, W_out, kD, kH, kW)
    strides = (
        input_padded.strides[0],
        input_padded.strides[1],
        input_padded.strides[2] * stride_d,
        input_padded.strides[3] * stride_h,
        input_padded.strides[4] * stride_w,
        input_padded.strides[2],
        input_padded.strides[3],
        input_padded.strides[4],
    )
    windows = np.lib.stride_tricks.as_strided(input_padded, shape=shape, strides=strides)
    col = windows.transpose(0, 1, 5, 6, 7, 2, 3, 4).reshape(
        B, C_in * kD * kH * kW, D_out * H_out * W_out
    )
    return col


def conv3d_ref(input, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0)):
    """Reference Conv3d using im2col + matmul.

    Args:
        input:   (B, C_in, D, H, W) float array
        weight:  (C_out, C_in, kD, kH, kW) float array
        bias:    (C_out,) float array or None
        stride:  (stride_d, stride_h, stride_w)
        padding: (pad_d, pad_h, pad_w) symmetric padding

    Returns:
        output: (B, C_out, D_out, H_out, W_out) float array
    """
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    B, C_in, D, H, W = input.shape
    C_out, _, kD, kH, kW = weight.shape

    D_out = (D + 2 * pad_d - kD) // stride_d + 1
    H_out = (H + 2 * pad_h - kH) // stride_h + 1
    W_out = (W + 2 * pad_w - kW) // stride_w + 1

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        input_padded = np.pad(
            input,
            ((0, 0), (0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
        )
    else:
        input_padded = np.ascontiguousarray(input)

    if not input_padded.flags["C_CONTIGUOUS"]:
        input_padded = np.ascontiguousarray(input_padded)

    col = im2col_3d(input_padded, kD, kH, kW, stride_d, stride_h, stride_w, D_out, H_out, W_out)

    weight_mat = weight.reshape(C_out, -1)  # (C_out, C_in*kD*kH*kW)
    output = np.matmul(weight_mat, col)  # (B, C_out, D_out*H_out*W_out)
    output = output.reshape(B, C_out, D_out, H_out, W_out)

    if bias is not None:
        output = output + bias.reshape(1, -1, 1, 1, 1)

    return output
