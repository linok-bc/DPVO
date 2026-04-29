#!/usr/bin/env python3
"""Mask out a center region from all images in a folder."""

import argparse
import cv2
from pathlib import Path
from tqdm import tqdm


def mask_images(input_dir: Path, output_dir: Path,
                width_frac: float, height_frac: float,
                center_x_frac: float, center_y_frac: float,
                fill_value: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg"}
    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in exts)
    if not images:
        raise FileNotFoundError(f"No images found in {input_dir}")

    for img_path in tqdm(images, desc="Masking"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Skipping unreadable file: {img_path.name}")
            continue

        h, w = img.shape[:2]
        mask_w = int(w * width_frac)
        mask_h = int(h * height_frac)
        cx = int(w * center_x_frac)
        cy = int(h * center_y_frac)

        x0 = max(cx - mask_w // 2, 0)
        y0 = max(cy - mask_h // 2, 0)
        x1 = min(x0 + mask_w, w)
        y1 = min(y0 + mask_h, h)

        img[y0:y1, x0:x1] = fill_value
        cv2.imwrite(str(output_dir / img_path.name), img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask center region of images.")
    parser.add_argument("input_dir", type=Path, help="Folder containing input images")
    parser.add_argument("output_dir", type=Path, help="Folder to save masked images")
    parser.add_argument("--width_frac",  type=float, default=0.30, help="Mask width as fraction of image width")
    parser.add_argument("--height_frac", type=float, default=0.50, help="Mask height as fraction of image height")
    parser.add_argument("--center_x_frac", type=float, default=0.50, help="Mask center x (fraction)")
    parser.add_argument("--center_y_frac", type=float, default=0.65, help="Mask center y (fraction; >0.5 = lower half)")
    parser.add_argument("--fill", type=int, default=0, help="Fill value (0=black, 255=white)")
    args = parser.parse_args()

    mask_images(args.input_dir, args.output_dir,
                args.width_frac, args.height_frac,
                args.center_x_frac, args.center_y_frac, args.fill)
    print(f"Done. Masked images in {args.output_dir}")
