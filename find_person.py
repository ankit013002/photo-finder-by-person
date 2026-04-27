#!/usr/bin/env python3
"""
Person Finder — scans a photo/video album and copies every file
containing a specific person's face into an output directory.

Uses DeepFace (ArcFace model) — no dlib / Visual Studio required.

Usage:
  python find_person.py --refs ./grandma_pics --album ./family_album --output ./found
  python find_person.py --refs ./grandma_pics --album ./family_album --output ./found --tolerance 0.55 --dry-run
"""

import os
import sys
import shutil
import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # suppress TensorFlow noise

try:
    from deepface import DeepFace
except ImportError:
    print("Missing dependency: pip install --user deepface")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Missing dependency: pip install --user pillow")
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
    ".tiff", ".tif",
    *([".heic", ".heif"] if HEIC_SUPPORTED else []),
}

VIDEO_EXTS = {
    ".mp4", ".mov", ".avi", ".mkv", ".m4v",
    ".wmv", ".flv", ".3gp", ".mts", ".m2ts", ".ts",
}

MODEL_NAME     = "ArcFace"
DETECTOR       = "retinaface"   # accurate; fall back to "opencv" if slow

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Face embedding helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.array(a), np.array(b)
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def get_embeddings(img_path: str) -> List[np.ndarray]:
    """Return a list of face embeddings found in img_path (one per detected face)."""
    try:
        results = DeepFace.represent(
            img_path=img_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=True,
        )
        return [np.array(r["embedding"]) for r in results]
    except Exception:
        return []


def embeddings_match(
    candidate_embs: List[np.ndarray],
    reference_embs: List[np.ndarray],
    tolerance: float,
) -> bool:
    """Return True if any candidate face is close enough to any reference face."""
    for cand in candidate_embs:
        for ref in reference_embs:
            if _cosine_distance(cand, ref) <= tolerance:
                return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  Reference loading
# ─────────────────────────────────────────────────────────────────────────────

def load_reference_embeddings(ref_dir: str) -> List[np.ndarray]:
    path = Path(ref_dir)
    if not path.is_dir():
        raise ValueError(f"Reference directory not found: {ref_dir}")

    candidates = [
        f for f in path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    ]
    if not candidates:
        raise ValueError(f"No images found in reference directory: {ref_dir}")

    log.info(f"Loading {len(candidates)} reference image(s) with {MODEL_NAME} ...")
    log.info("(First run downloads the model weights — ~500 MB, one-time only)")

    all_embs: List[np.ndarray] = []
    for img_path in candidates:
        embs = get_embeddings(str(img_path))
        if not embs:
            log.warning(f"  No face detected in: {img_path.name}")
            continue
        if len(embs) > 1:
            log.warning(f"  Multiple faces in {img_path.name} — using the first one")
        all_embs.append(embs[0])
        log.info(f"  OK  {img_path.name}")

    if not all_embs:
        raise ValueError(
            "Could not extract any face embeddings from reference images.\n"
            "Make sure the images clearly show the person's face."
        )

    log.info(f"Loaded {len(all_embs)} reference embedding(s).")
    return all_embs


# ─────────────────────────────────────────────────────────────────────────────
#  Per-file checks
# ─────────────────────────────────────────────────────────────────────────────

def check_image(img_path: Path, ref_embs: List[np.ndarray], tolerance: float) -> bool:
    embs = get_embeddings(str(img_path))
    return embeddings_match(embs, ref_embs, tolerance)


def check_video(
    vid_path: Path,
    ref_embs: List[np.ndarray],
    tolerance: float,
    sample_secs: int,
) -> bool:
    """Sample one frame every `sample_secs` seconds; return True on first match."""
    try:
        import cv2
    except ImportError:
        log.warning("opencv-python not installed — skipping video (pip install --user opencv-python)")
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

        # Write frame to a temp file so DeepFace can read it
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            import cv2 as _cv2
            _cv2.imwrite(tmp_path, frame)
            embs = get_embeddings(tmp_path)
            if embeddings_match(embs, ref_embs, tolerance):
                found = True
        finally:
            os.unlink(tmp_path)

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
    ref_embs = load_reference_embeddings(ref_dir)

    log.info(f"Collecting media from: {album_dir}")
    all_files = list(collect_media(album_dir))
    images = [f for f in all_files if f.suffix.lower() in IMAGE_EXTS]
    videos = [f for f in all_files if f.suffix.lower() in VIDEO_EXTS]
    log.info(f"Found {len(images)} image(s) and {len(videos)} video(s).")

    out_path = Path(output_dir)
    matched, errors = [], []

    # ── Images ────────────────────────────────────────────────────────────────
    if images:
        log.info("Scanning images ...")
        for img in tqdm(images, desc="Images", unit="img", disable=not HAS_TQDM):
            try:
                if check_image(img, ref_embs, tolerance):
                    matched.append(img)
                    if not dry_run:
                        safe_copy(img, out_path)
            except Exception as exc:
                log.debug(f"Error ({img.name}): {exc}")
                errors.append(img)

    # ── Videos ────────────────────────────────────────────────────────────────
    if videos:
        log.info(f"Scanning videos (1 frame every {video_sample_secs}s) ...")
        for vid in tqdm(videos, desc="Videos", unit="vid", disable=not HAS_TQDM):
            try:
                if check_video(vid, ref_embs, tolerance, video_sample_secs):
                    matched.append(vid)
                    if not dry_run:
                        safe_copy(vid, out_path)
            except Exception as exc:
                log.debug(f"Error ({vid.name}): {exc}")
                errors.append(vid)

    # ── Summary ───────────────────────────────────────────────────────────────
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
        prog="find_person",
        description="Find every photo/video in an album containing a specific person.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python find_person.py --refs ./grandma_pics --album ./family_album --output ./grandma_found
  python find_person.py --refs ./grandma_pics --album ./family_album --output ./grandma_found --tolerance 0.45
  python find_person.py --refs ./grandma_pics --album ./family_album --output ./grandma_found --dry-run

Tolerance guide (cosine distance — lower = stricter)
-----------------------------------------------------
  0.35  very strict  — fewest false positives, may miss older/blurry photos
  0.45  recommended  — good balance (default)
  0.55  lenient      — catches more, small risk of false positives

Tips
----
  • Use 5–15 clear, varied reference photos (different ages, angles, lighting)
  • The first run downloads ArcFace model weights (~500 MB) — subsequent runs are instant
  • Use --dry-run first to preview results before copying thousands of files
        """,
    )
    parser.add_argument("--refs",   required=True, metavar="DIR",
                        help="Folder with clear reference photos of the person to find")
    parser.add_argument("--album",  required=True, metavar="DIR",
                        help="Photo/video album folder to search (searched recursively)")
    parser.add_argument("--output", required=True, metavar="DIR",
                        help="Destination folder for matched files")
    parser.add_argument("--tolerance", type=float, default=0.45, metavar="N",
                        help="Cosine distance threshold 0.0–1.0 (lower = stricter). Default: 0.45")
    parser.add_argument("--video-sample", type=int, default=30, metavar="SECS",
                        help="Sample one video frame every N seconds. Default: 30")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show matches without copying any files")
    parser.add_argument("--verbose", action="store_true",
                        help="Show debug-level output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not HEIC_SUPPORTED:
        log.info("Tip: install pillow-heif for iPhone HEIC support: pip install --user pillow-heif")

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
