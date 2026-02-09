import torch
import triton
import triton.language as tl
DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')


def naive_softmax(x):
    '''
    Built for input of size (M,N)
    Safe softmax is when we subtract the maximum element in order to avoid numerical 
    overflows when doing .exp(); softmax is invariant to this shift
    '''
    # read MN elements, find their max along N, and write M elements (the maxes)
    x_max = x.max(dim=1)[0] 
        # pytorch actually outputs a tuple of (values, indices) so [0] grabs the values;
        # we ignored the indices when talking about memory writes above
    # read MN + M elements, subtraction is MN flops, and write MN elements
    z = x - x_max[:, None]
    # read MN elements and write MN elemnts
    numerator = torch.exp(z)
        # exp is actually a lot of flops per element but we're only worried about mem ops rn
    # read MN elements, do MN flops to find M sum values, and then write M elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements, division is MN flops, then write MN elements
    out = numerator / denominator[:, None]

    # in total we did 8MN + 4M memory operations
    # (read 5MN + 2M elements; wrote 3MN + 2M elements)
    return out


properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
# each Streaming Multi-processor (SM) is like a mini-processor that can run multiple programs
NUM_SM = properties["multiprocessor_count"] 
# registers are the fastest memory on the GPU
NUM_REGS = properties["max_num_regs"] 
    # each SM has a limited number of registers; 
    # programs share these registers, so using too many per program limits parallelism
# each SM has a dedicated pool of SRAM that it can access
# since there can be multiple programs per SM, those programs share the same SRAM
    # ^that will be very useful information later in the matmul tutorial
TOTAL_SRAM_PER_SM = properties["max_shared_mem"] 
# a warp is a group of threads that execute together
# a thread can be thought of as analagous to a single CPU core, but far more limited in the operations it can do
WARP_SIZE = properties["warpSize"]# usually 32 on nvidia GPUs and 64 on AMD


def softmax(x):
    assert x.ndim == 2
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)


    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    # number of actions to do together
    num_stages = 4 if TOTAL_SRAM_PER_SM > 200_000 else 2

    y = torch.empty_like(x)

    #warm up to precompute the kernel :

    kernel = _softmax_kernel.warmup(
        x,y,
        n_rows,n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=(1,)
    )

    kernel._init_handles()
    n_regs = kernel.n_regs
    sram_needed_per_program = kernelmetada


### sadly forgot to git push and lost the complete code, feel very sad but that is lif

