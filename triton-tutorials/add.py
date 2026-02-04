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
    n_elements = output.numel() # gives the number of elements in a tensor
    grid = lambda meta_parameters: (triton.cdiv(n_elements,meta["BLOCK_SIZE"]))  # needs to be a tuple of entries and these needs to be integers

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




