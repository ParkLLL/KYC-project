from __future__ import annotations

from pathlib import Path

from PIL import Image
import numpy as np


def split_cards(
    source_image: str | Path,
    output_dir: str | Path = "data/processed/cards",
) -> list[Path]:
    src = Path(source_image)
    if not src.exists():
        raise FileNotFoundError(f"Source image not found: {src}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(src).convert("RGB")
    arr = np.array(img)

    # Detect the main row where cards are present (white background excluded).
    mask = np.any(arr < 245, axis=2)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("Could not detect cards in image.")
    y0, y1 = int(ys.min()), int(ys.max())

    # Detect 3 horizontal card segments by column density.
    col_density = mask[y0 : y1 + 1, :].sum(axis=0)
    active = np.where(col_density > 30)[0]
    runs: list[tuple[int, int]] = []
    start = int(active[0])
    prev = int(active[0])
    for v in active[1:]:
        cur = int(v)
        if cur == prev + 1:
            prev = cur
            continue
        runs.append((start, prev))
        start = cur
        prev = cur
    runs.append((start, prev))

    if len(runs) < 3:
        raise ValueError(f"Expected 3 card regions, got {len(runs)}")

    # Keep the 3 widest detected runs.
    runs = sorted(runs, key=lambda r: r[1] - r[0], reverse=True)[:3]
    runs = sorted(runs, key=lambda r: r[0])

    outputs: list[Path] = []
    for idx, (x0, x1) in enumerate(runs, start=1):
        # Add a small margin while staying in bounds.
        mx, my = 8, 6
        left = max(0, x0 - mx)
        top = max(0, y0 - my)
        right = min(arr.shape[1] - 1, x1 + mx)
        bottom = min(arr.shape[0] - 1, y1 + my)
        card = img.crop((left, top, right + 1, bottom + 1))
        out_file = out_dir / f"IC{idx:03d}.png"
        card.save(out_file)
        outputs.append(out_file)

    return outputs


if __name__ == "__main__":
    files = split_cards("data/raw/Hypothetical images of ICs.png")
    print("Created card images:")
    for p in files:
        print(f"- {p}")
