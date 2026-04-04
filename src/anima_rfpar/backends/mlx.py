from dataclasses import dataclass


@dataclass
class MlxBackend:
    """Placeholder MLX backend contract for Apple Silicon parity work."""

    name: str = "mlx"

    def is_available(self) -> bool:
        try:
            import mlx.core  # type: ignore

            return True
        except Exception:
            return False

    def describe(self) -> str:
        return "MLX backend scaffold (Metal op integration TODO)."
