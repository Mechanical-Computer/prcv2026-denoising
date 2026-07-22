"""Prepare submission package for PRCV2026 Challenge.

Usage:
    python src/prepare_submit.py --result_dir results/val
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path


TEAM_ID = "PRCV2026-0025"
TEAM_NAME = "TinyLight"


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare submission zip")
    parser.add_argument(
        "--result_dir", type=str, default="results/val",
        help="Directory containing denoised result images.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="submit",
        help="Directory to save the submission zip.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect result images
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    images = sorted(
        [p for p in result_dir.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=lambda p: p.name,
    )
    print(f"Found {len(images)} result images in {result_dir}")

    if len(images) == 0:
        print("ERROR: No result images found! Run inference first.")
        return 1

    # Create team_info.txt content
    team_info_content = f"team_id: {TEAM_ID}\nteam_name: {TEAM_NAME}\n"

    # Create zip file
    zip_name = f"Results_{TEAM_NAME}.zip"
    zip_path = output_dir / zip_name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add team_info.txt at root level
        zf.writestr("team_info.txt", team_info_content)

        # Add all result images at root level (no subdirectory!)
        for img_path in images:
            zf.write(img_path, img_path.name)
            
    file_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\n[OK] Submission package created: {zip_path}")
    print(f"  Size: {file_size_mb:.1f} MB")
    print(f"  Images: {len(images)}")
    print(f"  Team ID: {TEAM_ID}")
    print(f"  Team Name: {TEAM_NAME}")

    # Validate the zip structure
    print("\n--- Validating zip structure ---")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        has_team_info = "team_info.txt" in names
        has_subdirs = any("/" in n for n in names)
        image_count = sum(1 for n in names if n.endswith(".png"))

        print(f"  team_info.txt at root: {'[OK]' if has_team_info else '[ERROR] MISSING!'}")
        print(f"  No subdirectories: {'[OK]' if not has_subdirs else '[ERROR] HAS SUBDIRS!'}")
        print(f"  Image count: {image_count}")

        if has_team_info and not has_subdirs:
            print("\n[OK] Submission package is valid!")
        else:
            print("\n[ERROR] Submission package has issues, please fix!")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
