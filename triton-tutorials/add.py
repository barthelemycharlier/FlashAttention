import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

def add(x,y):
    """
    """
    #pre-allocate the output
    output = torch.empty_like(x)
    # check tensors are on same device
    assert x.device == y.device and  x.device ==DEVICE, "Input tensors must be on the same device"

    # define our launche grid, number of kernel instances or number of programs
    # that will run in parallel
    # code runs hundreds of time in parallel accross the SM on the gpu
    # number of element in our tensor
    n_elements = output.numel() # gives the number of elements in a tensor # could use x.numel() as well
    grid = lambda meta_parameters: (triton.cdiv(n_elements,meta_parameters["BLOCK_SIZE"]),)  # needs to be a tuple of entries and these needs to be integers

    # here that tuple is (4,), cdvi(m,n) = (m+(n-1))//n
    # if n_element is not cleanly divislible then cdiv handles it ? 

    #here we need to index that function, we need to subset it ? using our launching grid
    add_kernel[grid](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE=1024
    )
    return output


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_element : int,
    BLOCK_SIZE: tl.constexpr): 
    """
    the decorator tells python to compile it on gpu,

    the inputs are pointers, not full tensors to the memory to the first entry of 
    x, y and output,

    BLOCK_SIZE will be regarded as an already compiled constant at compile time, meaning
    #TODO explain it better

    """
    #there is multiple instance of the kernel running in parallel,
    #to see which chunk of the actual data is it mine to calculate

    PID = tl.program_id(axis=0) # 0 referring to the grid (4,) so PID is going to be either 0 one two or 3
    # vector of lenght 256 and blocksize of size 64
    #PID 0 might process elements [0:64]
    #PID 1 might process elements [64:128]

    block_start = PID*BLOCK_SIZE 

    offsets = block_start + tl.arange(0,BLOCK_SIZE) # array of int32 of lenght block_size
    mask = offsets < n_element 

    #load data from DRAM/VRAM/HBM to SRAM/on-chip memory

    x = tl.load(x_ptr + offsets, mask=mask,other = None) # shape (BLOCK_SIZE)
    y = tl.load(y_ptr + offsets, mask=mask,other = None) # default is None so is not a problem

    output = x + y 

    #write data back to dram 

    tl.store(output_ptr+offsets, value=output ,mask=mask)

    ## the problem is that the kernel has a bug,
    ## we need to account for potentially number of elements not divisible by blocksize, 
    # thus we create a mask  




    



def test_add_kernel(size,atol=1e-3,rtol=1e-3):
    """
    size
    absolute tolerance
    relative tol
    """
    torch.manual_seed(42)

    # create two tensors 
    x = torch.randn(size, device=DEVICE,dtype=torch.float32)
    y = torch.randn(size, device=DEVICE,dtype=torch.float32)

    #run python kernel and pytorch equivalent 
    z_tri = add(x,y) ## method we are going to build

    z_ref = x + y # pytorch's way

    #compare, assert_close gives percentages on errors 
    torch.testing.assert_close(
        z_tri, z_ref, atol=atol, rtol=rtol
    )
    print("no errors")
 
# how fast is it compared to pytorch

# create benchmark

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12,28,1)], # if you have errors like how much value
        # of the DRAM is allocated then you can lower 28 to 20 or 22
        x_log=True,
        line_arg='provider',
        line_vals=['triton','torch'],
        line_names=["Triton","Torch"],
        styles=[('blue','-'),('green','-')],
        ylabel='GB/s', #  ususally performance is measures in how much you can read/write
        plot_name ="vec-add-perf",
        args={}
        

        )
    )

def benchmark(size,provider):
    """
    This function benchmarks the performance of vector addition for different providers.
    Depending on the 'provider' argument ('triton' or 'torch'), it performs addition
    using the Triton kernel or PyTorch's built-in addition, respectively. This helps
    compare the speed and memory throughput of custom Triton code versus PyTorch.
    """
    # create input data
    # create two tensors 
    x = torch.randn(size, device=DEVICE,dtype=torch.float32)
    y = torch.randn(size, device=DEVICE,dtype=torch.float32)

    quantiles = [0.5,0.05,0.95]

    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)

    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    else:
        raise ValueError("Unknown provider: {}".format(provider))

    ## we did 3 memory operations (load twice and wrote once), times number of elements times "float32"
    ## then convert to Gb per second
    gbps = lambda ms: 3 * x.numel() * x.element_size() / (ms * 1e-3) / 1e9
    return gbps(ms), gbps(max_ms), gbps(min_ms)







if __name__ == "__main__":
    print("doing additions")
    test_add_kernel(size=40)
    test_add_kernel(size=89475)
    test_add_kernel(size=1775)

    import sys
    
    if len(sys.argv)>1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=True)

