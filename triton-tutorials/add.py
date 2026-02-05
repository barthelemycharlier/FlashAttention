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
    BLOCK_SIZE: tl.constexpr,) 
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



def test_add_kernel(size,atol=1e-3,rtol=1e-3):
    """
    size
    absolute tolerance
    relative tol
    """
    torch.manual_seed(42)

    # create two tensors 
    x = torch.randn(size, device=DEVICE)
    y = torch.randn(size, device=DEVICE)

    #run python kernel and pytorch equivalent 
    z_tri = add(x,y) ## method we are going to build

    z_ref = x + y # pytorch's way

    #compare, assert_close gives percentages on errors 
    torch.testing.assert_close(
        z_tri, z_ref, atol=atol, rtol=rtol
    )
    print("no errors")




