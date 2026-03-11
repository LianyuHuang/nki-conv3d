# nki-conv3d

The first [NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html) Conv3d kernel for AWS Trainium.

## Why

Video generation models (Wan2.1/2.2, CogVideoX, HunyuanVideo) use 3D VAEs built on `Conv3d` / `CausalConv3d`. AWS Trainium has no native Conv3d support — the NKI ecosystem only has [Conv1d](https://github.com/aws-neuron/nki-library/tree/main/src/nkilib_src/nkilib/experimental/conv). This gap blocks **all** video generation models from running on Trainium.

This repo fills that gap.

| NKI Kernel | Exists Before This Repo |
|------------|------------------------|
| Conv1d     | ✅ (nki-library)        |
| Conv2d     | ❌                      |
| **Conv3d** | ❌ → **✅ this repo**   |

## Algorithm

Conv3d is decomposed into temporal slices of Conv2d, each computed via im2col + GEMM:

```
output[:, :, d, :, :] = Σ_{kd} Conv2d(input[:, :, d*s+kd, :, :], weight[:, :, kd, :, :])
```

This decomposition is **exact** (not an approximation). The GEMM leverages the NeuronCore matrix multiplication engine via `nisa.nc_matmul`.

## Files

| File | Description |
|------|-------------|
| `conv3d.py` | NKI kernel — the main deliverable |
| `conv3d_ref.py` | NumPy reference (im2col + matmul) for testing |
| `test_conv3d.py` | 35+ test cases across 3 layers |

## Test Coverage

| Layer | Cases | Source |
|-------|-------|--------|
| PyTorch standard | 12 | Adapted from `torch/testing/_internal/common_nn.py` |
| Wan2.1 VAE configs | 12 | Actual CausalConv3d shapes from `wan/modules/vae.py` |
| Edge cases | 7+ | Single channel, D=1, mixed strides, causal padding |

All tests compare against `torch.nn.functional.conv3d` as ground truth.

## Quick Start

### Run tests (NumPy reference only, no NKI needed)

```bash
pip install numpy pytest torch
pytest test_conv3d.py -k "Ref" -v
```

### Run NKI kernel tests (requires neuronxcc)

```bash
pip install neuronxcc  # or install NeuronSDK
pytest test_conv3d.py -k "NKI" -v
```

The NKI kernel runs via `nki.baremetal` (CPU simulation) — **no Trainium hardware required** for development and testing.

### Use in your model

```python
import nki
from conv3d import conv3d

# CPU simulation
result = nki.baremetal(conv3d)(input_np, weight_np, bias_np,
                                stride_d=1, stride_h=1, stride_w=1,
                                pad_d=1, pad_h=1, pad_w=1)
```

## CausalConv3d (Wan2.1/2.2 VAE)

Wan's `CausalConv3d` applies asymmetric temporal padding `(2*pad, 0)` **before** calling standard Conv3d. This kernel handles the Conv3d part; causal padding is done at the Python wrapper level:

```python
import numpy as np
from conv3d_ref import conv3d_ref

# Simulate CausalConv3d(3,3,3) with padding=(1,1,1)
input_causal = np.pad(input, ((0,0), (0,0), (2,0), (1,1), (1,1)), mode="constant")
output = conv3d_ref(input_causal, weight, stride=(1,1,1), padding=(0,0,0))
```

## Roadmap

- [x] NumPy reference implementation with im2col + matmul
- [x] NKI kernel with temporal decomposition + nc_matmul
- [x] Comprehensive test suite (35+ cases)
- [x] Wan2.1 VAE CausalConv3d compatibility tests
- [ ] Bulk DMA loading for im2col (replace element-wise loads)
- [ ] Performance benchmarks on trn1/trn2
- [ ] bfloat16 precision tests
- [ ] Backward pass (for training)
- [ ] PR to [aws-neuron/nki-library](https://github.com/aws-neuron/nki-library)

## Related

- [aws-neuron/nki-library](https://github.com/aws-neuron/nki-library) — Official NKI kernels (Conv1d, Flash Attention, RoPE, RMSNorm)
- [aws-neuron/nki-samples](https://github.com/aws-neuron/nki-samples) — NKI tutorials and examples
- [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) — Video generation model whose 3D VAE needs this kernel
- [neuronx-distributed-inference #57](https://github.com/aws-neuron/neuronx-distributed-inference/pull/57) — LTX-2 video model on Trainium (DiT only, no Conv3D)

## License

Apache 2.0
