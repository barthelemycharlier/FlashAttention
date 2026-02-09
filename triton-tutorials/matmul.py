### idea is to do :

"""
A@B=C
(M,K) @ (K,N) = (M,N)
for m in range(0,M):
    for n in range(0,N):
        a_vec = A[m,:]
        b_vec = B[:,n]
        C[m, n] = dot(a_vec,b_vec)

Matrix version 

A@B = C
(M,K) @ (K,N) = (M,N)
 
for m in range(0,M,BLOCK_SIZE): #in parallel, each iteration is its own PID
    for m in range(0,N,BLOCK_SIZE): #in parallel, each iteration is its own PID
        acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
        for k in range(0, K, BLOCK_SIZE):
            # Load the next BLOCK_SIZE chunk of A and B from global memory into SRAM
            a_tile = A[m : m + BLOCK_SIZE, k : k + BLOCK_SIZE]
            b_tile = B[k : k + BLOCK_SIZE, n : n + BLOCK_SIZE]
            # Accumulate the partial results
            acc += tl.dot(a_tile, b_tile)
        # Write result back to C
        C[m : m + BLOCK_SIZE, n : n + BLOCK_SIZE] = acc


"""
import torch
import triton
import triton.language as tl
DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')


# actual kernel for matmul

#! 
#? 
#* We have been using arbitrary heuristics for the BLOCK_SIZE_M, BLOCK_SIZE_N
#* we can have triton autotuning :

# autotuning is just setting up a bunch of different potential meta-parameters configurations that Triton will automatically
# choose from later based on which one performs best on our specific GPU. Triton will figure out for us which one to use. They're 
# all values chosen heuristically, but notice everything is a multiple of 32 in sticking w/ the number of threads in a warp.
autotune_configs = [ #* meta parameter 
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2)
]
# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator which consumes
#   1) a list of `triton.Config` objects that define different configs of meta-parameters and compilation options
#   2) an auto-tuning *key* whose change in values will trigger a new evaluation of all the provided configs, meaning
#       that any time either M, N, or K changes with a new input, Triton will check which config is best all over again

@triton.autotune(configs = autotune_configs, key = ['M','N','K'])
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M,N,K,
    stride_a_M, stride_a_K,
    stride_b_K,stride_b_N,
    stride_c_M,stride_c_N,
    #meta-parameters #? defautl values precized in the autoconfig part
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """
    """
    PID = tl.program_id(axis=0)
    num_PID_along_M = tl.cdiv(M,BLOCK_SIZE_M)
    num_PID_along_N = tl.cdiv(N,BLOCK_SIZE_N)
    num_PID_in_group = GROUP_SIZE * num_PID_along_N  #! defining as columns of B ?
    group_id = PID // num_PID_in_group
    first_PID_in_group_along_M = group



"""
to debug, you can use prints and all

import os 
os.environ["TRITON_INTERNPRET"] = "1"
"""


def matmul(a,b):

    assert a.ndim == b.ndim = 2
    assert a.shape[1] == b.shape[0]

    (M,K), (_,N) = a.shape, b.shape

    c = torch.empty((M,N), device = a.device, dtype=torch.float16)

    """
    launch grid fro now :
    [0, 1, 2, 3]
    [4, 5, 6, 7]
    [8,9,10, 11 ]
    [12,13,14,15]
    """
    grid = lambda meta: (
        triton.cdiv(M,meta['BLOCK_SIZE_M']) * triton.cdiv(N,meta['BLOCK_SIZE_N'])
    )# (16,)
    # cdvi -> one above 

    _matmul_kernel[grid](
        a,b,c,
        M,N,K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )




def test_matmul_kernel(size:tuple, atol = 1e-2,rtol = 1e-1,device=DEVICE):
    """
    those tolerance values should be what you get for a complex kernel
    """
    torch.manual_seed(0)
    asset type(size)==tuple and len(size) == 2
    a = torch.randn(size, device=DEVICE, dtype = torch.float16)
    b = torch.randn(size, device=DEVICE, dtype = torch.float16)
    c_tri = matmul(a,b)
    c_ref = torch.matmul(a,b)
    torch.testing.assert_close(c_tri, c_ref, atol=atol, rtol=rtol)


