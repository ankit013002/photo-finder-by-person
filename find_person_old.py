#!/usr/bin/env python3
"""
Person Finder (dlib / face_recognition version) — archived.

Requires dlib, which needs cmake + Visual C++ Build Tools to compile on Windows.
Use find_person.py instead (DeepFace/ArcFace, no build tools needed).

Install requirements (Windows):
  1. Install CMake: https://cmake.org/download/
  2. Install Visual C++ Build Tools:
       https://visualstudio.microsoft.com/visual-cpp-build-tools/
  3. pip install cmake dlib face-recognition Pillow opencv-python tqdm pillow-heif

Usage:
  python find_person_old.py --refs ./grandma_pics --album ./family_album --output ./found
  python find_person_old.py --refs ./grandma_pics --album ./family_album --output ./found --tolerance 0.45 --dry-run
"""

import os
import sys
import shutil
import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np

try:
    import face_recognition
except ImportError:
    print("Missing dependency: pip install face-recognition")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Missing dependency: pip install Pillow")
    sys.exit(1)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(it, **kw):
        return it

# ── HEIC / HEIF support (iPhone photos) ──────────────────────────────────────
HEIC_SUPPORTED = False
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:
    pass

# ── File extension sets ───────────────────────────────────────────────────────
IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    ".tiff", ".tif", ".gif",
    *([".heic", ".heif"] if HEIC_SUPPORTED else []),
}

VIDEO_EXTS = {
    ".mp4", ".mov", ".avi", ".mkv", ".m4v",
    ".wmv", ".flv", ".3gp", ".mts", ".m2ts", ".ts",
}

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_reference_encodings(ref_dir: str) -> List[np.ndarray]:
    path = Path(ref_dir)
    if not path.is_dir():
        raise ValueError(f"Reference directory not found: {ref_dir}")

    candidates = [
        f for f in path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    ]
    if not candidates:
        raise ValueError(f"No images found in reference directory: {ref_dir}")

    log.info(f"Loading {len(candidates)} reference image(s) ...")
    encodings: List[np.ndarray] = []

    for img_path in candidates:
        try:
            img = face_recognition.load_image_file(str(img_path))
            encs = face_recognition.face_encodings(img, num_jitters=2)
            if not encs:
                log.warning(f"  No face detected in: {img_path.name}")
                continue
            if len(encs) > 1:
                log.warning(f"  Multiple faces in {img_path.name} — using the first one")
            encodings.append(encs[0])
            log.info(f"  OK  {img_path.name}")
        except Exception as exc:
            log.warning(f"  SKIP  {img_path.name}: {exc}")

    if not encodings:
        raise ValueError(
            "Could not extract any face encodings from reference images.\n"
            "Make sure the images clearly show the person's face."
        )

    log.info(f"Loaded {len(encodings)} reference encoding(s).")
    return encodings


def faces_match(
    candidate_encs: List[np.ndarray],
    reference_encs: List[np.ndarray],
    tolerance: float,
) -> bool:
    for enc in candidate_encs:
        if any(face_recognition.compare_faces(reference_encs, enc, tolerance=tolerance)):
            return True
    return False


def check_image(img_path: Path, reference_encs: List[np.ndarray], tolerance: float) -> bool:
    try:
        img = face_recognition.load_image_file(str(img_path))
        encs = face_recognition.face_encodings(img)
        return faces_match(encs, reference_encs, tolerance)
    except Exception as exc:
        log.debug(f"Image error ({img_path.name}): {exc}")
        return False


def check_video(
    vid_path: Path,
    reference_encs: List[np.ndarray],
    tolerance: float,
    sample_secs: int,
) -> bool:
    try:
        import cv2
    except ImportError:
        log.warning("opencv-python not installed — skipping video files")
        return False

    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        return False

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, int(fps * sample_secs))
    found = False
    idx   = 0

    while idx < total and not found:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb)
        if faces_match(encs, reference_encs, tolerance):
            found = True
        idx += step

    cap.release()
    return found


# ─────────────────────────────────────────────────────────────────────────────
#  File utilities
# ─────────────────────────────────────────────────────────────────────────────

def collect_media(album_dir: str):
    root = Path(album_dir)
    if not root.is_dir():
        raise ValueError(f"Album directory not found: {album_dir}")
    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS:
            yield f


def safe_copy(src: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        i = 1
        while dest.exists():
            dest = dest_dir / f"{src.stem}_{i}{src.suffix}"
            i += 1
    shutil.copy2(src, dest)
    return dest


# ─────────────────────────────────────────────────────────────────────────────
#  Main scan
# ─────────────────────────────────────────────────────────────────────────────

def run(
    ref_dir: str,
    album_dir: str,
    output_dir: str,
    tolerance: float,
    video_sample_secs: int,
    dry_run: bool,
):
    reference_encs = load_reference_encodings(ref_dir)

    log.info(f"Collecting media from: {album_dir}")
    all_files = list(collect_media(album_dir))
    images = [f for f in all_files if f.suffix.lower() in IMAGE_EXTS]
    videos = [f for f in all_files if f.suffix.lower() in VIDEO_EXTS]
    log.info(f"Found {len(images)} image(s) and {len(videos)} video(s).")

    out_path = Path(output_dir)
    matched, errors = [], []

    if images:
        log.info("Scanning images ...")
        for img in tqdm(images, desc="Images", unit="img", disable=not HAS_TQDM):
            try:
                if check_image(img, reference_encs, tolerance):
                    matched.append(img)
                    if not dry_run:
                        safe_copy(img, out_path)
            except Exception as exc:
                log.debug(f"Error ({img.name}): {exc}")
                errors.append(img)

    if videos:
        log.info(f"Scanning videos (1 frame every {video_sample_secs}s) ...")
        for vid in tqdm(videos, desc="Videos", unit="vid", disable=not HAS_TQDM):
            try:
                if check_video(vid, reference_encs, tolerance, video_sample_secs):
                    matched.append(vid)
                    if not dry_run:
                        safe_copy(vid, out_path)
            except Exception as exc:
                log.debug(f"Error ({vid.name}): {exc}")
                errors.append(vid)

    print(f"\n{'─'*52}")
    print(f"  SCAN COMPLETE")
    print(f"{'─'*52}")
    print(f"  Files scanned : {len(all_files)}")
    print(f"  Matched       : {len(matched)}")
    print(f"  Errors        : {len(errors)}")
    if dry_run:
        print(f"  (dry-run — nothing was copied)")
    else:
        print(f"  Output dir    : {output_dir}")
    print(f"{'─'*52}\n")

    if matched:
        print("Matched files:")
        for f in matched:
            print(f"  {f}")

    if errors:
        print("\nCould not process:")
        for f in errors:
            print(f"  {f}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="find_person_old",
        description="Find every photo/video containing a specific person (dlib version).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
NOTE: This version requires dlib + Visual C++ Build Tools.
      Use find_person.py (DeepFace) instead — it works without any build tools.

Tolerance guide (L2 distance — lower = stricter)
-------------------------------------------------
  0.4  very strict
  0.5  recommended (default)
  0.6  lenient
        """,
    )
    parser.add_argument("--refs",   required=True, metavar="DIR")
    parser.add_argument("--album",  required=True, metavar="DIR")
    parser.add_argument("--output", required=True, metavar="DIR")
    parser.add_argument("--tolerance",    type=float, default=0.5, metavar="N")
    parser.add_argument("--video-sample", type=int,   default=30,  metavar="SECS")
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--verbose",  action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.dry_run:
        log.info("DRY-RUN — no files will be copied.")
    if not (0.0 < args.tolerance <= 1.0):
        parser.error("--tolerance must be between 0.0 and 1.0")

    try:
        run(
            ref_dir=args.refs,
            album_dir=args.album,
            output_dir=args.output,
            tolerance=args.tolerance,
            video_sample_secs=args.video_sample,
            dry_run=args.dry_run,
        )
    except ValueError as exc:
        log.error(str(exc))
        sys.exit(1)
    except KeyboardInterrupt:
        log.info("Interrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
