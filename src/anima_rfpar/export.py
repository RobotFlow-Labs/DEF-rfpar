"""Export RFPAR agent model: pth -> safetensors -> ONNX -> TRT."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import torch

from .agent import REINFORCEAgent

logger = logging.getLogger(__name__)


def load_agent_from_checkpoint(
    ckpt_path: Path,
    img_h: int = 224,
    img_w: int = 224,
    channels: int = 3,
    detector_mode: bool = False,
) -> REINFORCEAgent:
    # weights_only=False required: checkpoint contains config dicts alongside state_dict
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    agent = REINFORCEAgent(
        img_h=img_h, img_w=img_w, channels=channels, detector_mode=detector_mode
    )
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()
    return agent


def export_safetensors(agent: REINFORCEAgent, output_path: Path) -> Path:
    from safetensors.torch import save_file

    state = agent.state_dict()
    st_path = output_path / "agent.safetensors"
    save_file(state, str(st_path))
    logger.info(f"Exported safetensors: {st_path}")
    return st_path


def export_onnx(
    agent: REINFORCEAgent, output_path: Path, img_h: int = 224, img_w: int = 224
) -> Path:
    import onnx

    onnx_path = output_path / "agent.onnx"
    dummy = torch.randn(1, 3, img_h, img_w)
    torch.onnx.export(
        agent,
        dummy,
        str(onnx_path),
        input_names=["image"],
        output_names=["action_mean", "action_std"],
        opset_version=18,
    )
    # Ensure weights are embedded (not external) for TRT compatibility
    model = onnx.load(str(onnx_path))
    onnx.save(model, str(onnx_path), save_as_external_data=False)
    logger.info(f"Exported ONNX: {onnx_path}")
    return onnx_path


def export_trt(onnx_path: Path, output_path: Path) -> tuple[Path | None, Path | None]:
    """Export TensorRT engines (FP16 + FP32) using shared toolkit."""
    trt_toolkit = Path("/mnt/forge-data/shared_infra/trt_toolkit/export_to_trt.py")
    fp16_path = None
    fp32_path = None

    if trt_toolkit.exists():
        import subprocess

        for precision in ["fp16", "fp32"]:
            out = output_path / f"agent_{precision}.trt"
            cmd = [
                "python",
                str(trt_toolkit),
                str(onnx_path),
                "--output",
                str(out),
                "--precision",
                precision,
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
                logger.info(f"Exported TRT {precision}: {out}")
                if precision == "fp16":
                    fp16_path = out
                else:
                    fp32_path = out
            except (
                subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired
            ) as e:
                logger.warning(f"TRT {precision} export failed: {e}")
                # Fallback: try trtexec directly
                try:
                    trtexec_cmd = [
                        "trtexec",
                        f"--onnx={onnx_path}",
                        f"--saveEngine={out}",
                    ]
                    if precision == "fp16":
                        trtexec_cmd.append("--fp16")
                    subprocess.run(
                        trtexec_cmd, check=True, capture_output=True, text=True, timeout=300
                    )
                    logger.info(f"Exported TRT {precision} via trtexec: {out}")
                    if precision == "fp16":
                        fp16_path = out
                    else:
                        fp32_path = out
                except Exception as e2:
                    logger.warning(f"trtexec {precision} also failed: {e2}")
    else:
        logger.warning(f"TRT toolkit not found at {trt_toolkit}, trying trtexec directly")
        import subprocess

        for precision in ["fp16", "fp32"]:
            out = output_path / f"agent_{precision}.trt"
            trtexec_cmd = ["trtexec", f"--onnx={onnx_path}", f"--saveEngine={out}"]
            if precision == "fp16":
                trtexec_cmd.append("--fp16")
            try:
                subprocess.run(
                    trtexec_cmd, check=True, capture_output=True, text=True, timeout=300
                )
                logger.info(f"Exported TRT {precision}: {out}")
                if precision == "fp16":
                    fp16_path = out
                else:
                    fp32_path = out
            except Exception as e:
                logger.warning(f"trtexec {precision} failed: {e}")

    return fp16_path, fp32_path


def export_all(
    ckpt_path: Path,
    output_dir: Path,
    detector_mode: bool = False,
) -> dict[str, Path | None]:
    """Run complete export pipeline: pth -> safetensors -> ONNX -> TRT."""
    output_dir.mkdir(parents=True, exist_ok=True)

    agent = load_agent_from_checkpoint(ckpt_path, detector_mode=detector_mode)

    # 1. Copy pth
    pth_dst = output_dir / "agent.pth"
    shutil.copy2(ckpt_path, pth_dst)
    logger.info(f"Copied pth: {pth_dst}")

    # 2. Safetensors
    st_path = export_safetensors(agent, output_dir)

    # 3. ONNX
    onnx_path = export_onnx(agent, output_dir)

    # 4. TRT FP16 + FP32
    fp16, fp32 = export_trt(onnx_path, output_dir)

    results = {
        "pth": pth_dst,
        "safetensors": st_path,
        "onnx": onnx_path,
        "trt_fp16": fp16,
        "trt_fp32": fp32,
    }

    with open(output_dir / "export_manifest.json", "w") as f:
        import json

        json.dump({k: str(v) if v else None for k, v in results.items()}, f, indent=2)

    logger.info(f"Export complete: {list(results.keys())}")
    return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Export RFPAR agent")
    parser.add_argument("checkpoint", type=Path, help="Path to agent checkpoint")
    parser.add_argument("--output", type=Path, default=Path("exports"), help="Output dir")
    parser.add_argument("--detector", action="store_true", help="Detection mode agent")
    args = parser.parse_args()

    export_all(args.checkpoint, args.output, detector_mode=args.detector)
