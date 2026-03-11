"""Minimal test for nc_matmul: stationary[K,M] @ moving[K,N] -> result[M,N]."""
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl


@nki.jit
def test_matmul_correct(a_t_ref, b_ref):
    """nc_matmul: stationary[K,M].T @ moving[K,N] -> [M,N]

    a_t_ref: [K, M] (transposed LHS)
    b_ref:   [K, N]
    result:  [M, N]
    """
    K, M = a_t_ref.shape
    K2, N = b_ref.shape

    a_sbuf = nl.load(a_t_ref)
    b_sbuf = nl.load(b_ref)

    acc = nl.zeros((M, N), dtype=nl.float32, buffer=nl.psum)
    acc += nisa.nc_matmul(a_sbuf, b_sbuf)

    result = nisa.tensor_copy(acc)
    out = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.hbm)
    nl.store(out, value=result)
    return out


if __name__ == "__main__":
    np.random.seed(42)
    M, K, N = 3, 12, 9
    a = np.random.randn(M, K).astype(np.float32)  # [M, K]
    b = np.random.randn(K, N).astype(np.float32)   # [K, N]
    expected = a @ b  # [M, N]

    # nc_matmul needs stationary[K, M], so pass a.T
    a_t = a.T.copy()  # [K, M]

    print("Test: nc_matmul with correct layout (stationary[K,M], moving[K,N])")
    try:
        result = nki.simulate_kernel(test_matmul_correct, a_t, b)
        print(f"  result shape: {result.shape}")
        print(f"  max diff: {np.max(np.abs(result - expected)):.6e}")
        print(f"  {'PASS' if np.allclose(result, expected, rtol=1e-4) else 'FAIL'}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
