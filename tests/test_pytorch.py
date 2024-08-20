"""
Basic tests for PyTorch and CUDA availability.
"""

import torch

def test_cuda_availability():
    """Test if CUDA is available."""
    assert torch.cuda.is_available(), "CUDA is not available"

def test_basic_torch_operation():
    """Test a basic PyTorch operation on GPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(3, 3).to(device)
    y = torch.randn(3, 3).to(device)
    z = torch.matmul(x, y)
    assert z.shape == (3, 3), f"Expected shape (3, 3), but got {z.shape}"
