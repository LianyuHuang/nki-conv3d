# NKI Kernel Development: Lessons Learned

## 1. API Signatures (neuronx-cc 2.x)

All NKI ISA operations are **functional** — they return new tiles, NOT mutate in-place.

### nisa.memset
```python
# WRONG: nisa.memset(tensor, 0)
# WRONG: nisa.memset(tensor, 0, dtype=nl.float32)
# RIGHT:
tile = nisa.memset((shape_tuple), value, dtype=nl.float32)
```
- Signature: `(shape, value, dtype, *, mask, engine) -> local_tile`
- Creates AND returns a new tile. Does NOT take a `dst` parameter.

### nisa.nc_matmul
```python
# WRONG: nisa.nc_matmul(dst=acc, stationary=w, moving=im)
# RIGHT:
result = nisa.nc_matmul(stationary, moving)
acc += result  # accumulate via +=
```
- Signature: `(stationary, moving, *, is_stationary_onezero, ...) -> local_tile`
- Returns a new tile. No `dst` parameter.
- **CRITICAL**: Computes `stationary.T @ moving`, NOT `stationary @ moving`.

### nisa.tensor_copy
```python
# WRONG: nisa.tensor_copy(dst=result, src=acc)
# RIGHT:
result = nisa.tensor_copy(src_tile)
```
- Signature: `(src, *, mask, dtype, engine) -> local_tile`
- Returns a new tile. No `dst` parameter.

### nl.zeros
```python
tile = nl.zeros((rows, cols), dtype=nl.float32, buffer=nl.psum)
```
- Works for both `nl.sbuf` and `nl.psum` buffers.
- Preferred over `nisa.memset` for zero-initialization.

### nl.load / nl.store
```python
sbuf_tile = nl.load(hbm_ref[indices])          # HBM -> SBUF, returns tile
nl.store(hbm_ref[indices], value=sbuf_tile)    # SBUF -> HBM, returns None
```

## 2. nc_matmul Layout: stationary.T @ moving

This was the biggest gotcha. `nc_matmul` computes **stationary.T @ moving**.

Both inputs must have the **contraction dimension K as the FIRST (partition) dimension**:
```
stationary[K, M]  (K=partition, M=free, M≤128)
moving[K, N]      (K=partition, N=free, N≤512)
result = [M, N]   (stationary.T @ moving)
```

For Conv3d where we want `W[C_out, K] @ im2col[K, spatial]`:
```python
# Weight tile must be TRANSPOSED: [K, C_out] not [C_out, K]
w_tile = nl.zeros((col_tile, c_out_tile), ...)  # [K, M]
im2col = nl.zeros((col_tile, sp_tile), ...)     # [K, N]
acc += nisa.nc_matmul(w_tile, im2col)           # -> [M, N] = [C_out, spatial]
```

## 3. Python `if` Inside NKI Loops: DOES NOT WORK Reliably

### The problem
```python
# This FAILS in simulate_kernel when the condition is sometimes False:
for local_sp in nl.affine_range(sp_size):
    h_in = h_out * stride_h - pad_h + kh
    if 0 <= h_in < H:  # <-- BREAKS when padding causes this to be False
        im2col[local_col, local_sp] = nl.load(input_ref[...])
```
Error: `ValueError: cannot reshape array of size 0 into shape (1,1)`

### Why
The NKI trace compiler/simulator cannot handle conditional branching that skips
`nl.load` operations inside loops. When a condition is ALWAYS true (no padding),
it works because the compiler effectively removes the condition. When it's
sometimes false (with padding), the simulator crashes.

### The NKI-idiomatic solution (from official Conv1d kernel)
The official NKI Conv1d kernel handles padding by:
1. **Pre-zeroing** SBUF buffers with `nisa.memset(dst=buffer, value=0.0)`
2. **Computing valid ranges** at Python level (compile time)
3. Using **bulk `tensor_copy`** to copy only valid data into correct positions
4. **Never reading out-of-bounds** — clamp indices before loading

### Our practical solution
Pre-pad the input in a Python wrapper BEFORE calling the NKI kernel:
```python
def conv3d(input_np, weight_np, bias_np=None, stride=(1,1,1), padding=(0,0,0)):
    pd, ph, pw = padding
    if pd > 0 or ph > 0 or pw > 0:
        input_np = np.pad(input_np, ((0,0),(0,0),(pd,pd),(ph,ph),(pw,pw)))
    return nki.simulate_kernel(conv3d_kernel, input_np, weight_np, bias_np,
                                stride_d=stride[0], stride_h=stride[1], stride_w=stride[2])
```
This matches CausalConv3d's pattern (pad then conv) and eliminates all boundary
logic from the kernel.

## 4. Python `range` vs `nl.affine_range` vs `nl.sequential_range`

### `range(N)` (Python)
- Fully unrolled at trace time. Each iteration becomes separate traced code.
- Python `if/else/continue` works (evaluated at trace time with concrete values).
- **Problem**: The NKI compiler doesn't properly scope tensor allocations across
  unrolled iterations. If you create `im2col = nl.zeros(...)` inside the loop,
  the compiler may treat all iterations as accessing the SAME tensor.

### `nl.affine_range(N)`
- Tells compiler iterations are independent (can parallelize).
- Loop variable is used for indexing but may be symbolic.
- **Problem**: `nl.affine_range(variable_bound)` where the bound depends on an
  outer loop variable — the compiler may use the MAX bound across all outer
  iterations, causing OOB errors.

### `nl.sequential_range(N)`
- Iterations execute in order. Supports loop-carried dependencies.
- Still symbolic — same scoping issues as `nl.affine_range`.

### Guideline
- Use `range()` for outer loops where you need Python control flow (if/continue)
  AND the loop count is small (B, D_out, kD).
- Use `nl.sequential_range()` for loops with loop-carried dependencies
  (accumulation across tiles).
- Use `nl.affine_range()` for inner compute loops with FIXED bounds.
- **NEVER** use `nl.affine_range(variable_bound)` where the bound changes
  between outer loop iterations.

## 5. Tile Dimension Limits

| Buffer | Partition (1st dim) | Free (2nd dim) |
|--------|-------------------|----------------|
| SBUF   | ≤ 128 (P_MAX)    | flexible       |
| PSUM   | ≤ 128             | ≤ 512 (F_MAX)  |

- `nc_matmul`: contraction dim K ≤ 128 (partition of both inputs)
- `nc_matmul`: stationary free dim M ≤ 128 (PE columns)
- `nc_matmul`: moving free dim N ≤ 512

## 6. The Conv1d Kernel Pattern (Official Reference)

The official NKI Conv1d kernel (`aws-neuron/nki-library`) uses this architecture:

1. **Outer loops**: Plain Python `for ... in range(...)` — NOT `nl.affine_range`
2. **Padding**: Pre-zero buffers, compute valid ranges at Python level, bulk-copy
   only valid data using `nisa.tensor_copy` with computed slice ranges
3. **Im2col equivalent**: "K-replication" strategy — scatters input into a stacked
   format along the partition dimension for efficient matmul
4. **Data movement**: Bulk DMA (`nisa.dma_copy`) for HBM↔SBUF, NOT element-wise
   `nl.load` / `nl.store`
5. **Accumulation**: `acc += nisa.nc_matmul(stationary, moving)` with PSUM buffer
6. **PSUM → HBM**: `nisa.tensor_copy(src=psum_tile)` then `nisa.dma_copy`

### Key difference from our approach
Our kernel uses **element-wise scalar loads** (`nl.load(input_ref[b,c,d,h,w])`)
to build im2col. The official Conv1d uses **bulk tile operations** (DMA + tensor_copy).
Element-wise loads work for small inputs but have issues with:
- Boundary handling (can't use `if` guards)
- Variable loop bounds (compiler OOB errors)

## 7. Current Status and Open Issues

### What works (13/31 NKI tests pass — regression from earlier 22/31)
After revert, the baseline shows 13 passed, 18 failed.
The bias tests that previously passed now fail — this was NOT caused by
our tiling changes. It appears to be a pre-existing issue with the bias
code that was masked earlier or an environment change.

Passing tests: all non-bias standard tests, small-channel Wan VAE, non-bias edge cases.
Failing tests: ALL bias tests + large-channel Wan VAE tests.

### Two separate failure modes

**Mode 1: Bias tests — constant offset error (FIXED)**
- ALL tests with `use_bias=True` failed with a constant per-channel offset
- Root cause: bias loading used `nl.affine_range(c_out_tile)` with
  `min(co_start + local_co, C_out - 1)` — the `min()` on symbolic NKI
  variable produced wrong results (loaded wrong bias values).
- Fix: changed to `nl.affine_range(co_size)` with `bias_ref[co_start + local_co]`
  (matching the original committed version). **22/31 restored.**
- **Lesson: `min()` on symbolic NKI loop variables produces WRONG results,
  same as `%`. Never use Python built-in arithmetic guards on NKI symbolics.**

**Mode 2: Large-channel OOB — `nl.affine_range(variable_bound)`**
- Tests with `col_height > 128` OR `C_out > 128` fail with OOB errors
- Error: `IndexError: Out-of-bound access for tensor 'im2col' on dimension 4`
- Root cause: `nl.affine_range(col_size)` where col_size varies across outer loop iterations

### Failed fix attempts (2026-03 session 2)

**Attempt 1: `% col_height` wrapping with fixed `nl.affine_range(col_tile)`**
- Change: `flat_idx = (col_start + local_col) % col_height`
- Result: BROKE previously-passing non-bias tests (constant offset errors appeared)
- Analysis: `%` operator on symbolic NKI loop variables produces incorrect results.
  Even when mathematically equivalent to no-op (flat_idx < col_height), the
  symbolic `%` changes the compiler's data flow tracking and corrupts computation.
- **Lesson: Do NOT use `%` on symbolic NKI variables for index clamping.**

**Attempt 2: Python `range(col_start, col_end)` with concrete ints**
- Change: Replace `nl.affine_range(col_size)` with Python `range` for partition-dim
- Result: Same OOB for large channels (`[0, 1727] exceed dimension size of 128`),
  PLUS same bias regression as attempt 1.
- Analysis: The NKI compiler tracks `im2col[local_row, local_sp]` accesses across
  ALL unrolled Python iterations. Even though `local_row` is a concrete int (0-127
  per tile), the compiler merges tensor accesses across tile iterations and sees
  the full range.
- **Lesson: Python `range` unrolling does NOT fix the tensor scoping issue.
  The NKI compiler merges accesses to same-named tensors across unrolled iterations.**

### Findings from online research

1. **Official NKI matmul examples ALL require K % 128 == 0** — they assert this.
   No official example handles partial-tile contraction dimension.

2. **`nl.affine_range` requires ALL iterations to have the same tile shape.**
   Variable bounds cause the compiler to merge/vectorize ranges incorrectly.

3. **Recommended pattern: fixed-size tiles + `nl.arange` masking**
   ```python
   i_p = nl.arange(P_MAX)[:, None]
   valid_mask = (k_start + i_p) < contraction_dim
   tile = nl.load(tensor[k_start + i_p, :], mask=valid_mask)
   ```

4. **Alternative: pre-pad K on the host to a multiple of 128**
   Build im2col on the host, pad to K_padded = ceil(K/128)*128, pass as HBM tensor.

### Final solution: v2 architecture (31/31 PASS)

**Architecture**: Host-side im2col + tiled matmul NKI kernel.

1. **`conv3d()` wrapper** (Python/NumPy):
   - Pads input spatially/temporally
   - For each (b, d_out): builds im2col via `_build_im2col_2d_numpy` across all kD
   - Concatenates weight across kD into 2D matrix [C_out, K_total]
   - Pads K and C_out to multiples of 128
   - Calls `nki.simulate_kernel(tiled_matmul_kernel, w_padded, im_padded)`
   - Extracts valid region, adds bias, reshapes to 5D

2. **`tiled_matmul_kernel`** (NKI):
   - Receives pre-padded [K_padded, M_padded] and [K_padded, N]
   - ALL tile bounds are FIXED (128 for partition dim)
   - Uses `nl.arange` for bulk load/store (NOT element-wise nl.load)
   - Simple double loop: outer over M tiles, inner over K tiles
   - Accumulates in PSUM, copies to SBUF, stores to HBM

**Why this works**:
- Eliminates ALL variable-bound `nl.affine_range` from the NKI kernel
- K and M are padded to multiples of 128 on the host — every tile is exactly P_MAX
- Zero-padding contributes nothing to matmul (0 × weight = 0)
- Follows the official NKI matmul pattern exactly

**Key lesson**: The right abstraction boundary is crucial. Moving im2col to the host
and keeping the NKI kernel as a pure tiled matmul eliminates ALL the NKI compiler
issues (variable bounds, tensor scoping, symbolic arithmetic).

## 8. Development Environment

- `neuronxcc` only available on x86_64 Linux (no macOS/ARM64)
- Docker with `--platform linux/amd64` required on Mac
- `nki.simulate_kernel` runs on CPU (no Trainium hardware needed)
- `nki.baremetal` requires actual Neuron hardware
- Import: `import neuronxcc.nki as nki` (not `import nki`)

## 9. Key References

- [NKI Conv1d kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/conv/conv1d.py) — The closest reference implementation
- [NKI matmul tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/tutorials/matrix_multiplication.html)
- [nc_matmul API](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_matmul.html)
- [NKI programming model](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/programming_model.html)
- [NKI error reference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/nki.errors.html)
