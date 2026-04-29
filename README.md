# photo-finder

Scans a photo/video album and copies every file containing a specific person's face into an output directory.

Uses [DeepFace](https://github.com/serengil/deepface) with the ArcFace model — no dlib or Visual Studio required.

## Requirements

- Python 3.10+
- ~500 MB disk space for ArcFace model weights (downloaded automatically on first run)

## Installation

```bash
pip install -r requirements.txt
```

> **iPhone HEIC support** is included via `pillow-heif`. If you skip it, HEIC/HEIF files will be ignored.

## Usage

```bash
python find_person.py --refs <reference-dir> --album <album-dir> --output <output-dir> [options]
```

### Arguments

| Argument              | Required | Description                                                               |
| --------------------- | -------- | ------------------------------------------------------------------------- |
| `--refs DIR`          | Yes      | Folder containing clear reference photos of the person to find            |
| `--album DIR`         | Yes      | Photo/video album to search (searched recursively)                        |
| `--output DIR`        | Yes      | Destination folder for matched files                                      |
| `--tolerance N`       | No       | Cosine distance threshold `0.0`–`1.0` (lower = stricter). Default: `0.45` |
| `--video-sample SECS` | No       | Sample one video frame every N seconds. Default: `5`                      |
| `--dry-run`           | No       | Preview matches without copying any files                                 |
| `--verbose`           | No       | Show debug-level output                                                   |

### Examples

```bash
# Basic scan
python find_person.py --refs ./grandma_pics --album ./family_album --output ./grandma_found

# Stricter matching
python find_person.py --refs ./grandma_pics --album ./family_album --output ./grandma_found --tolerance 0.45

# Preview before copying
python find_person.py --refs ./grandma_pics --album ./family_album --output ./grandma_found --dry-run
```

## Tolerance guide

| Value  | Strictness  | Notes                                                   |
| ------ | ----------- | ------------------------------------------------------- |
| `0.35` | Very strict | Fewest false positives; may miss older or blurry photos |
| `0.45` | Recommended | Good balance (default)                                  |
| `0.55` | Lenient     | Catches more photos; small risk of false positives      |

## Supported formats

**Images:** JPEG, PNG, BMP, WebP, TIFF, HEIC/HEIF (iPhone)

**Videos:** MP4, MOV, AVI, MKV, M4V, WMV, FLV, 3GP, MTS, M2TS, TS

iPhone Live Photos (still + companion `.mov`) are handled automatically — when the still image matches, the companion video is copied alongside it.

## Tips

- Use **5–15 clear, varied reference photos** (different ages, angles, lighting) for best results.
- Run with `--dry-run` first to preview results before copying large libraries.
- The first run downloads ArcFace model weights (~500 MB to `~/.deepface/`). Subsequent runs are instant.
