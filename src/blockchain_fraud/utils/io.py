from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj: Any, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    return p


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_graph(graph: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    torch.save(graph, p)
    return p


def load_graph(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def write_table(df: pd.DataFrame, stem: str, output_dir: str | Path) -> None:
    out = ensure_dir(output_dir)
    df.to_csv(out / f"{stem}.csv", index=False)
    df.to_markdown(out / f"{stem}.md", index=False)
    df.to_latex(out / f"{stem}.tex", index=False, escape=True)

