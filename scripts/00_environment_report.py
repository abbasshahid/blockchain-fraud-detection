"""Record the exact software and hardware used for a run.

Reviewer 1 asked for library versions and complete hyperparameters to be
published so that the five-seed experiments can be reproduced independently.
This script writes a machine-generated environment record next to the metrics,
and a pinned requirements file, so neither has to be maintained by hand.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TRACKED = ["numpy", "pandas", "scikit-learn", "scipy", "torch", "xgboost", "pyyaml", "pydantic", "matplotlib", "tabulate"]


def package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for name in TRACKED:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not installed"
    return out


def hardware() -> dict[str, object]:
    import torch

    info: dict[str, object] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
    return info


def config_snapshot() -> dict[str, object]:
    import yaml

    configs: dict[str, object] = {}
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        with path.open("r", encoding="utf-8") as f:
            configs[path.name] = yaml.safe_load(f)
    return configs


def dataset_hashes() -> dict[str, str]:
    from blockchain_fraud.data.validate_raw import locate_raw_files
    from blockchain_fraud.utils.hashes import sha256_file

    try:
        files = locate_raw_files(ROOT / "elliptic_bitcoin_dataset")
    except Exception:
        return {}
    return {name: sha256_file(path) for name, path in files.items()}


def freeze_requirements(destination: Path) -> None:
    versions = package_versions()
    lines = [
        "# Pinned versions of the environment that produced the reported results.",
        "# Regenerate with: python scripts/00_environment_report.py",
        f"# python {platform.python_version()}",
    ]
    lines.extend(f"{name}=={value}" for name, value in versions.items() if value != "not installed")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def main() -> None:
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version,
        "python_version": platform.python_version(),
        "git_commit": git_commit(),
        "packages": package_versions(),
        "hardware": hardware(),
        "raw_dataset_sha256": dataset_hashes(),
        "configs": config_snapshot(),
        "seeds": [11, 22, 33, 44, 55],
    }
    out_dir = ROOT / "outputs" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "environment.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    freeze_requirements(ROOT / "requirements.lock.txt")
    print(f"Wrote {out_dir / 'environment.json'} and requirements.lock.txt")
    print(json.dumps({"packages": report["packages"], "hardware": report["hardware"]}, indent=2))


if __name__ == "__main__":
    main()
