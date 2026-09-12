from __future__ import annotations

import platform

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    requested = (requested or "auto").lower()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_report(device: torch.device) -> dict[str, str | int | bool]:
    report: dict[str, str | int | bool] = {
        "selected_device": str(device),
        "python_platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        report["cuda_device_name"] = torch.cuda.get_device_name(device)
        report["cuda_device_count"] = torch.cuda.device_count()
    return report

