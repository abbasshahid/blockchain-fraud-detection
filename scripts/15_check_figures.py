"""Verify that the paper panels are uniform and nothing is cropped.

Two defects are easy to introduce and hard to see in a thumbnail:

* panels that end up at different sizes, which makes a row of them look ragged
  once LaTeX scales each one to the same column width;
* content cut off by the canvas edge -- most often a rotated y-axis label that
  has grown longer than the axes it is centred on, or the outermost x-tick
  label overhanging the right edge.

This script checks both by rasterising each panel and looking for ink on the
outer border. Run it after regenerating the figures and before rebuilding the
manuscript.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fitz
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANELS = [
    "fig09a_threshold_sensitivity",
    "fig09b_reliability",
    "fig10a_evidence_coverage",
    "fig10b_deletion_curve",
]
BORDER_PX = 2
INK_THRESHOLD = 245


def check(fig_dir: Path, panels: list[str], dpi: int = 600) -> int:
    failures: list[str] = []
    sizes: set[tuple[float, float]] = set()

    for stem in panels:
        path = fig_dir / f"{stem}.pdf"
        if not path.exists():
            print(f"{stem:32s} MISSING ({path})")
            failures.append(stem)
            continue
        page = fitz.open(path)[0]
        sizes.add((round(page.rect.width, 2), round(page.rect.height, 2)))
        pixmap = page.get_pixmap(dpi=dpi)
        image = np.frombuffer(pixmap.samples, dtype=np.uint8)
        image = image.reshape(pixmap.height, pixmap.width, pixmap.n)[:, :, :3]
        ink = image.min(axis=2) < INK_THRESHOLD

        edges = {
            "top": ink[:BORDER_PX, :],
            "bottom": ink[-BORDER_PX:, :],
            "left": ink[:, :BORDER_PX],
            "right": ink[:, -BORDER_PX:],
        }
        touched = [name for name, band in edges.items() if band.any()]
        status = "CROPPED at " + ", ".join(touched) if touched else "clean"
        print(f"{stem:32s} {page.rect.width:6.1f} x {page.rect.height:5.1f} pt   {status}")
        if touched:
            failures.append(stem)

    print()
    if len(sizes) > 1:
        print(f"FAIL: panels use different canvases: {sorted(sizes)}")
        return 1
    if sizes:
        width, height = sizes.pop()
        print(f"All panels share one canvas: {width} x {height} pt")
    if failures:
        print(f"FAIL: {len(failures)} panel(s) crop their content or are missing")
        return 1
    print("No panel crops its content.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Check paper panels for uniform size and cropping.")
    parser.add_argument("--fig-dir", default=str(ROOT / "outputs" / "figures"))
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    sys.exit(check(Path(args.fig_dir), PANELS, args.dpi))


if __name__ == "__main__":
    main()
