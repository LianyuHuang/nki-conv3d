# Copyright 2026 Lianyu Huang
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Pure NumPy reference implementation of Conv3d for testing."""

import numpy as np


def im2col_3d(input_padded, kD, kH, kW, stride_d, stride_h, stride_w, D_out, H_out, W_out,
              dilation_d=1, dilation_h=1, dilation_w=1):
    """Convert 5D input to column matrix for Conv3d via matmul.

    Args:
        input_padded: (B, C_in, D_pad, H_pad, W_pad) padded input
        kD, kH, kW: kernel dimensions
        stride_d, stride_h, stride_w: strides
        D_out, H_out, W_out: output spatial dimensions
        dilation_d, dilation_h, dilation_w: dilation rates (default 1)

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
        input_padded.strides[2] * dilation_d,
        input_padded.strides[3] * dilation_h,
        input_padded.strides[4] * dilation_w,
    )
    windows = np.lib.stride_tricks.as_strided(input_padded, shape=shape, strides=strides)
    col = windows.transpose(0, 1, 5, 6, 7, 2, 3, 4).reshape(
        B, C_in * kD * kH * kW, D_out * H_out * W_out
    )
    return col


def conv3d_ref(input, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0),
               dilation=(1, 1, 1), groups=1):
    """Reference Conv3d using im2col + matmul.

    Args:
        input:    (B, C_in, D, H, W) float array
        weight:   (C_out, C_in/groups, kD, kH, kW) float array
        bias:     (C_out,) float array or None
        stride:   (stride_d, stride_h, stride_w)
        padding:  (pad_d, pad_h, pad_w) symmetric padding
        dilation: (dilation_d, dilation_h, dilation_w)
        groups:   number of groups for grouped convolution (default: 1)

    Returns:
        output: (B, C_out, D_out, H_out, W_out) float array
    """
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    dilation_d, dilation_h, dilation_w = dilation
    B, C_in, D, H, W = input.shape
    C_out = weight.shape[0]
    C_in_per_group = weight.shape[1]
    kD, kH, kW = weight.shape[2], weight.shape[3], weight.shape[4]

    assert C_in % groups == 0, (
        f"C_in ({C_in}) must be divisible by groups ({groups})"
    )
    assert C_out % groups == 0, (
        f"C_out ({C_out}) must be divisible by groups ({groups})"
    )

    C_out_per_group = C_out // groups

    # Effective kernel size accounts for dilation
    kD_eff = kD + (kD - 1) * (dilation_d - 1)
    kH_eff = kH + (kH - 1) * (dilation_h - 1)
    kW_eff = kW + (kW - 1) * (dilation_w - 1)

    D_out = (D + 2 * pad_d - kD_eff) // stride_d + 1
    H_out = (H + 2 * pad_h - kH_eff) // stride_h + 1
    W_out = (W + 2 * pad_w - kW_eff) // stride_w + 1

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

    # Allocate output
    output = np.zeros((B, C_out, D_out, H_out, W_out), dtype=input.dtype)

    # Process each group independently
    for g in range(groups):
        cin_start = g * C_in_per_group
        cin_end = cin_start + C_in_per_group
        cout_start = g * C_out_per_group
        cout_end = cout_start + C_out_per_group

        # Slice input channels for this group
        input_group = np.ascontiguousarray(input_padded[:, cin_start:cin_end])

        col = im2col_3d(
            input_group, kD, kH, kW,
            stride_d, stride_h, stride_w,
            D_out, H_out, W_out,
            dilation_d=dilation_d, dilation_h=dilation_h, dilation_w=dilation_w,
        )

        # Weight for this group: [C_out_per_group, C_in_per_group * kD * kH * kW]
        w_group = weight[cout_start:cout_end].reshape(C_out_per_group, -1)
        out_group = np.matmul(w_group, col)  # (B, C_out_per_group, D_out*H_out*W_out)
        output[:, cout_start:cout_end] = out_group.reshape(
            B, C_out_per_group, D_out, H_out, W_out
        )

    if bias is not None:
        output = output + bias.reshape(1, -1, 1, 1, 1)

    return output


def col2im_3d(col, input_shape, kD, kH, kW, stride_d, stride_h, stride_w,
              D_out, H_out, W_out, dilation_d=1, dilation_h=1, dilation_w=1):
    """Inverse of im2col_3d: scatter column matrix back to input tensor, accumulating overlaps.

    Args:
        col: (B, C_in * kD * kH * kW, D_out * H_out * W_out)
        input_shape: (B, C_in, D_pad, H_pad, W_pad) shape of the padded input
        kD, kH, kW: kernel dimensions
        stride_d, stride_h, stride_w: strides
        D_out, H_out, W_out: output spatial dimensions
        dilation_d, dilation_h, dilation_w: dilation rates

    Returns:
        result: (B, C_in, D_pad, H_pad, W_pad) accumulated gradients
    """
    B, C_in, D_pad, H_pad, W_pad = input_shape
    result = np.zeros(input_shape, dtype=col.dtype)

    col_reshaped = col.reshape(B, C_in, kD, kH, kW, D_out, H_out, W_out)

    for kd in range(kD):
        for kh in range(kH):
            for kw in range(kW):
                for d in range(D_out):
                    for h in range(H_out):
                        for w in range(W_out):
                            d_in = d * stride_d + kd * dilation_d
                            h_in = h * stride_h + kh * dilation_h
                            w_in = w * stride_w + kw * dilation_w
                            result[:, :, d_in, h_in, w_in] += col_reshaped[:, :, kd, kh, kw, d, h, w]

    return result


def conv3d_backward_ref(grad_output, input, weight, bias=None,
                        stride=(1, 1, 1), padding=(0, 0, 0),
                        dilation=(1, 1, 1), groups=1):
    """Backward pass for Conv3d, computing gradients w.r.t. input, weight, and bias.

    Args:
        grad_output: (B, C_out, D_out, H_out, W_out) gradient of loss w.r.t. output
        input:       (B, C_in, D, H, W) original input (before padding)
        weight:      (C_out, C_in/groups, kD, kH, kW) convolution weights
        bias:        (C_out,) or None
        stride:      (stride_d, stride_h, stride_w)
        padding:     (pad_d, pad_h, pad_w)
        dilation:    (dilation_d, dilation_h, dilation_w)
        groups:      number of groups

    Returns:
        grad_input:  (B, C_in, D, H, W) gradient w.r.t. input
        grad_weight: (C_out, C_in/groups, kD, kH, kW) gradient w.r.t. weight
        grad_bias:   (C_out,) gradient w.r.t. bias, or None if bias is None
    """
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    dilation_d, dilation_h, dilation_w = dilation
    B, C_in, D, H, W = input.shape
    C_out = weight.shape[0]
    C_in_per_group = weight.shape[1]
    kD, kH, kW = weight.shape[2], weight.shape[3], weight.shape[4]
    C_out_per_group = C_out // groups

    # Effective kernel size
    kD_eff = kD + (kD - 1) * (dilation_d - 1)
    kH_eff = kH + (kH - 1) * (dilation_h - 1)
    kW_eff = kW + (kW - 1) * (dilation_w - 1)

    D_out = (D + 2 * pad_d - kD_eff) // stride_d + 1
    H_out = (H + 2 * pad_h - kH_eff) // stride_h + 1
    W_out = (W + 2 * pad_w - kW_eff) // stride_w + 1

    # Pad input
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

    _, _, D_pad, H_pad, W_pad = input_padded.shape

    # 1. grad_bias: sum over batch and spatial dimensions
    grad_bias = None
    if bias is not None:
        grad_bias = grad_output.sum(axis=(0, 2, 3, 4))  # [C_out]

    # 2. grad_weight and grad_input per group
    grad_weight = np.zeros_like(weight)
    grad_input_padded = np.zeros_like(input_padded)

    for g in range(groups):
        cin_start = g * C_in_per_group
        cin_end = cin_start + C_in_per_group
        cout_start = g * C_out_per_group
        cout_end = cout_start + C_out_per_group

        # im2col of input for this group
        input_group = np.ascontiguousarray(input_padded[:, cin_start:cin_end])
        col = im2col_3d(
            input_group, kD, kH, kW,
            stride_d, stride_h, stride_w,
            D_out, H_out, W_out,
            dilation_d=dilation_d, dilation_h=dilation_h, dilation_w=dilation_w,
        )  # [B, C_in_per_group * kD * kH * kW, D_out * H_out * W_out]

        # grad_output for this group: [B, C_out_per_group, D_out * H_out * W_out]
        go_group = grad_output[:, cout_start:cout_end].reshape(
            B, C_out_per_group, D_out * H_out * W_out
        )

        # grad_weight: sum over batch of (grad_output @ col.T)
        # go_group: [B, C_out_per_group, spatial]
        # col:      [B, K, spatial]  where K = C_in_per_group * kD * kH * kW
        # grad_w = go_group @ col.T -> [B, C_out_per_group, K]
        grad_w = np.matmul(go_group, col.transpose(0, 2, 1))  # [B, C_out_per_group, K]
        grad_w = grad_w.sum(axis=0)  # [C_out_per_group, K]
        grad_weight[cout_start:cout_end] = grad_w.reshape(
            C_out_per_group, C_in_per_group, kD, kH, kW
        )

        # grad_input: col2im(weight.T @ grad_output)
        # w_group: [C_out_per_group, K]
        w_group = weight[cout_start:cout_end].reshape(C_out_per_group, -1)
        # grad_col = w_group.T @ go_group -> [B, K, spatial]
        grad_col = np.matmul(w_group.T, go_group)  # [B, K, spatial]

        # col2im: scatter grad_col back to input shape
        grad_input_group = col2im_3d(
            grad_col,
            (B, C_in_per_group, D_pad, H_pad, W_pad),
            kD, kH, kW,
            stride_d, stride_h, stride_w,
            D_out, H_out, W_out,
            dilation_d=dilation_d, dilation_h=dilation_h, dilation_w=dilation_w,
        )
        grad_input_padded[:, cin_start:cin_end] += grad_input_group

    # Remove padding from grad_input
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        grad_input = grad_input_padded[
            :, :,
            pad_d:(D_pad - pad_d) if pad_d > 0 else D_pad,
            pad_h:(H_pad - pad_h) if pad_h > 0 else H_pad,
            pad_w:(W_pad - pad_w) if pad_w > 0 else W_pad,
        ]
    else:
        grad_input = grad_input_padded

    return grad_input, grad_weight, grad_bias
