"""Backend contracts for CUDA/MLX/CPU implementations."""

from .cuda import CudaBackend
from .mlx import MlxBackend

__all__ = ["CudaBackend", "MlxBackend"]
