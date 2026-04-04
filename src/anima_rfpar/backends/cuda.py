from dataclasses import dataclass


@dataclass
class CudaBackend:
    """Placeholder CUDA backend contract for upcoming kernel integration."""

    name: str = "cuda"

    def is_available(self) -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def describe(self) -> str:
        return "CUDA backend scaffold (kernel binding TODO)."
