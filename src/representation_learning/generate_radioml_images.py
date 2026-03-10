#!/usr/bin/env python3
"""Generate constellation-diagram PNG images from RadioML .npy signals.

Usage:
    python -m src.representation_learning.generate_radioml_images \
        --data_root /path/to/RadioML \
        --snr_levels snr_0db snr_-10db snr_10db snr_-20db snr_20db

For each .npy signal file under  {snr}/train/{Class}/  and  {snr}/test/{Class}/,
a sibling directory  {snr}/train/img/  /  {snr}/test/img/
is created containing the corresponding .png constellation diagram.

A constellation diagram is a scatter plot of the in-phase (I) vs quadrature (Q)
components of the signal — the same representation used by the "own" dataset's
``noisyImg/`` directory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


# ── Constellation diagram rendering ─────────────────────────────────────

IMG_SIZE = 224   # match the "own" dataset's constellation images


def signal_to_constellation_image(
    signal_data: np.ndarray,
    size: int = IMG_SIZE,
    dot_radius: int = 1,
    bg_color: int = 0,
    fg_color: int = 255,
) -> Image.Image:
    """Render an I/Q signal as a constellation diagram (scatter plot).

    Parameters
    ----------
    signal_data
        ``(N,)`` complex or ``(N, 2)`` float array with I/Q samples.
    size
        Output image width/height in pixels.
    dot_radius
        Radius (px) of each constellation point.
    bg_color / fg_color
        Background and foreground greyscale values (0-255).

    Returns
    -------
    PIL.Image.Image
        ``(size, size)`` RGB image.
    """
    # Normalise to complex 1-D
    if signal_data.ndim == 2 and signal_data.shape[1] == 2:
        i_vals = signal_data[:, 0].astype(np.float64)
        q_vals = signal_data[:, 1].astype(np.float64)
    elif signal_data.ndim == 2 and signal_data.shape[0] == 2:
        i_vals = signal_data[0].astype(np.float64)
        q_vals = signal_data[1].astype(np.float64)
    elif np.iscomplexobj(signal_data):
        i_vals = signal_data.real.astype(np.float64)
        q_vals = signal_data.imag.astype(np.float64)
    else:
        raise ValueError(
            f"Cannot interpret signal shape {signal_data.shape} as I/Q data"
        )

    # Map I/Q to pixel coordinates  (centre the scatter, uniform scaling)
    margin = 0.05  # 5 % border
    all_vals = np.concatenate([i_vals, q_vals])
    vmin, vmax = all_vals.min(), all_vals.max()
    span = vmax - vmin
    if span < 1e-12:
        span = 1.0   # degenerate (constant) signal

    def _to_px(vals: np.ndarray) -> np.ndarray:
        normed = (vals - vmin) / span                 # [0, 1]
        return (margin + normed * (1 - 2 * margin)) * (size - 1)

    x_px = _to_px(i_vals)
    y_px = _to_px(q_vals)

    # Draw on a greyscale canvas, then convert to RGB
    img = Image.new("L", (size, size), color=bg_color)
    draw = ImageDraw.Draw(img)
    for x, y in zip(x_px, y_px):
        draw.ellipse(
            [x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius],
            fill=fg_color,
        )
    return img.convert("RGB")


# ── Batch generation ─────────────────────────────────────────────────────

def generate_images_for_split(
    split_dir: str,
    img_subdir: str = "img",
    overwrite: bool = False,
) -> int:
    """Generate constellation PNGs for every .npy under *split_dir*.

    Creates ``{split_dir}/img/{Class}_{stem}.png`` for each
    ``{split_dir}/{Class}/{stem}.npy`` signal file.

    Returns the number of images generated.
    """
    split_path = Path(split_dir)
    if not split_path.exists():
        return 0

    npy_files = sorted(split_path.rglob("*.npy"))
    generated = 0

    for npy_path in tqdm(npy_files, desc=f"  {split_path.name}", unit="img"):
        class_dir = npy_path.parent
        img_dir = split_path / img_subdir
        img_dir.mkdir(parents=True, exist_ok=True)

        class_name = class_dir.name
        img_name = f"{class_name}_{npy_path.stem}.png"
        img_path = img_dir / img_name

        if img_path.exists() and not overwrite:
            continue

        signal = np.load(str(npy_path))
        img = signal_to_constellation_image(signal)
        img.save(str(img_path))
        generated += 1

    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True,
                        help="Root directory containing snr_*db/ folders")
    parser.add_argument("--snr_levels", nargs="+", default=None,
                        help="SNR folders to process (default: all snr_*db)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-generate existing images")
    args = parser.parse_args()

    root = Path(args.data_root)
    if args.snr_levels:
        snr_dirs = [root / s for s in args.snr_levels]
    else:
        snr_dirs = sorted(root.glob("snr_*"))

    total = 0
    for snr_dir in snr_dirs:
        print(f"\n=== {snr_dir.name} ===")
        for split in ("train", "test"):
            n = generate_images_for_split(
                str(snr_dir / split),
                overwrite=args.overwrite,
            )
            total += n
            print(f"    {split}: {n} new images")

    print(f"\nDone — {total} constellation images generated in total.")


if __name__ == "__main__":
    main()
