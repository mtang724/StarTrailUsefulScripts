#!/usr/bin/env python3
"""
Star-trail Polaris alignment (translation-only).

Workflow:
- Solve ONLY the first frame (astrometry.net) to get Polaris pixel (x0,y0) in that reference frame.
- Build a small template around Polaris from the first frame.
- For each frame: template-match in a local ROI around (x0,y0) → (dx,dy), score.
- Apply translation (-dx,-dy) to align Polaris.
- Compute common crop using ONLY good frames (score >= threshold) so bad frames don't shrink to 1×1.
- Save aligned, cropped frames as true 16-bit PNGs (OpenCV) to frames_aligned/.
- Optionally copy low-score frames to frames_bad/.

Requires: numpy, rawpy, opencv-python(-headless), astropy, astroquery, imageio
"""

import os, csv, shutil, tempfile
from pathlib import Path
import numpy as np
import cv2
import rawpy
import imageio.v3 as iio

from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
from astroquery.astrometry_net import AstrometryNet

# ========= CONFIG =========
INPUT_DIR      = "frames_in"        # Put RAF/JPG/PNG/TIF/FITS here
ALIGNED_DIR    = "frames_aligned"   # Output aligned + cropped (16-bit PNG)
BAD_DIR        = "frames_bad"       # Low-score frames (optional)
LOG_PATH       = "polar_align_log.csv"

EXTS = (".raf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".fits")

# Astrometry.net (first frame only)
ASTROMETRY_API_KEY = "PUT_YOUR_KEY_HERE"   # get at https://nova.astrometry.net
SOLVE_TIMEOUT_S    = 120
FALLBACK_ON_SOLVE_FAIL = True  # Use brightest-near-center if solving fails

# Matching parameters
TEMPLATE_HALF   = 16      # template radius around Polaris in ref frame (→ 33×33)
ROI_HALF        = 160     # search half-size per frame (keep large if drifted)
MIN_MATCH_SCORE = 0.50    # frames below this are treated as "bad" (excluded)

# Safety: if final crop is tiny, stop and warn
MIN_CROP_WH     = 64      # minimum acceptable cropped width/height in pixels

# Polaris (ICRS)
POLARIS = SkyCoord('02h31m49.09s', '+89d15m50.8s', unit=(u.hourangle, u.deg), frame='icrs')
# ==========================


def ensure_dir(d: str):
    Path(d).mkdir(parents=True, exist_ok=True)


def list_frames() -> list[Path]:
    ps = [Path(INPUT_DIR) / n for n in sorted(os.listdir(INPUT_DIR))
          if n.lower().endswith(EXTS)]
    if not ps:
        raise RuntimeError("No frames in INPUT_DIR")
    return ps


def read_gray_float(path: Path) -> np.ndarray:
    """
    Return grayscale float32 in ~[0,1], geometry unchanged.
    Handles RAF/JPG/PNG/TIFF/FITS.
    """
    ext = path.suffix.lower()
    if ext == ".raf":
        with rawpy.imread(str(path)) as raw:
            rgb16 = raw.postprocess(no_auto_bright=True, use_camera_wb=False,
                                    output_bps=16, gamma=(1, 1), half_size=False)
        g = cv2.cvtColor((rgb16.astype(np.float32) / 65535.0), cv2.COLOR_RGB2GRAY)
        return g
    elif ext in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
        img = iio.imread(str(path))
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.astype(np.float32)
        # normalize conservatively
        if img.max() > 1.0:
            img = img / (65535.0 if img.max() > 255.0 else 255.0)
        return np.clip(img, 0, 1)
    elif ext == ".fits":
        from astropy.io import fits
        with fits.open(str(path)) as hdul:
            data = hdul[0].data.astype(np.float32)
        p99 = np.percentile(data, 99.5)
        return np.clip(data / p99, 0, 1) if p99 > 0 else data
    else:
        raise RuntimeError(f"Unsupported: {ext}")


def read_color16(path: Path) -> np.ndarray:
    """
    Return H×W×3 uint16 RGB (linear-ish) for writing.
    """
    ext = path.suffix.lower()
    if ext == ".raf":
        with rawpy.imread(str(path)) as raw:
            rgb16 = raw.postprocess(no_auto_bright=True, use_camera_wb=False,
                                    output_bps=16, gamma=(1, 1), half_size=False)
        return rgb16.astype(np.uint16)  # RGB, 0..65535
    else:
        img = iio.imread(str(path))
        if img.ndim == 2:  # gray → 3ch
            img = np.stack([img] * 3, axis=-1)
        if img.dtype == np.uint8:
            return (img.astype(np.uint16) * 257)  # 8→16
        elif img.dtype == np.uint16:
            return img
        else:
            # float -> scale to 16-bit
            arr = img.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / arr.max()
            arr = np.clip(arr, 0, 1)
            return (arr * 65535.0 + 0.5).astype(np.uint16)


def save_temp_tiff_for_solve(path: Path) -> Path:
    """
    Create a linear 16-bit TIFF (single-channel) for the solver.
    """
    g = read_gray_float(path)
    tiff16 = (np.clip(g, 0, 1) * 65535).astype(np.uint16)
    tmp = Path(tempfile.gettempdir()) / f"_solve_{path.stem}.tiff"
    iio.imwrite(str(tmp), tiff16)
    return tmp


def solve_ref_get_polaris_xy(first_path: Path):
    """
    Solve first image to get Polaris pixel. Fallback: brightest near center.
    Returns (x0, y0, ref_gray)
    """
    print(f"Solving first frame: {first_path.name}")
    tmp = save_temp_tiff_for_solve(first_path)
    ast = AstrometryNet()
    ast.api_key = ASTROMETRY_API_KEY

    hdr = None
    try:
        hdr = ast.solve_from_image(str(tmp),
                                   force_image_upload=True,
                                   solve_timeout=SOLVE_TIMEOUT_S)
    except Exception as e:
        print("Solve error:", e)

    if hdr is None:
        if not FALLBACK_ON_SOLVE_FAIL:
            raise RuntimeError("Solve failed and fallback disabled.")
        print("Fallback: brightest-near-center.")
        g = read_gray_float(first_path)
        h, w = g.shape
        cx, cy = w // 2, h // 2
        frac = 0.35
        dx, dy = int(w * frac / 2), int(h * frac / 2)
        x1, x2 = max(0, cx - dx), min(w, cx + dx)
        y1, y2 = max(0, cy - dy), min(h, cy + dy)
        crop = cv2.GaussianBlur(g[y1:y2, x1:x2], (0, 0), 1.2)
        _, _, _, maxLoc = cv2.minMaxLoc(crop)
        px, py = x1 + maxLoc[0], y1 + maxLoc[1]
        return float(px), float(py), g
    else:
        wcs = WCS(hdr)
        x, y = wcs.world_to_pixel(POLARIS)
        return float(x), float(y), read_gray_float(first_path)


def extract_template(img: np.ndarray, center_xy, half: int) -> np.ndarray:
    x, y = map(int, np.round(center_xy))
    h, w = img.shape
    x1, x2 = x - half, x + half + 1
    y1, y2 = y - half, y + half + 1
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        raise ValueError("Template out of bounds; reduce TEMPLATE_HALF or check location.")
    patch = img[y1:y2, x1:x2].astype(np.float32)
    m, s = patch.mean(), patch.std()
    return (patch - m) / s if s > 1e-6 else patch * 0


def match_shift(img: np.ndarray, ref_xy, tmpl: np.ndarray, roi_half: int):
    """
    Template-match in a local ROI around ref_xy.
    Returns dx, dy, score, (peak_x, peak_y)
    """
    cx, cy = ref_xy
    h, w = img.shape
    x1, x2 = max(0, int(cx - roi_half)), min(w, int(cx + roi_half + 1))
    y1, y2 = max(0, int(cy - roi_half)), min(h, int(cy + roi_half + 1))
    roi = img[y1:y2, x1:x2].astype(np.float32)
    if roi.shape[0] < tmpl.shape[0] or roi.shape[1] < tmpl.shape[1]:
        raise ValueError("ROI smaller than template; increase ROI_HALF or reduce TEMPLATE_HALF.")

    res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
    _, maxVal, _, maxLoc = cv2.minMaxLoc(res)
    th, tw = tmpl.shape
    peak_x = x1 + maxLoc[0] + tw // 2
    peak_y = y1 + maxLoc[1] + th // 2
    dx = float(peak_x - cx)
    dy = float(peak_y - cy)
    return dx, dy, float(maxVal), (float(peak_x), float(peak_y))


def compute_common_crop(w: int, h: int, tx_list: list[float], ty_list: list[float]):
    """
    For translation (tx,ty) applied to each frame:
      Valid x_out satisfies tx <= x_out < W + tx  (so that x_in = x_out - tx is in [0,W))
    Intersection across frames:
      x1 = ceil(max(tx_i)),  x2 = floor(min(W + tx_i))
      y1 = ceil(max(ty_i)),  y2 = floor(min(H + ty_i))
    Return integer crop box (x1,y1,x2,y2), with bounds clamped into image.
    """
    x1 = int(np.ceil(max(tx_list))) if tx_list else 0
    x2 = int(np.floor(min([w + t for t in tx_list]))) if tx_list else w
    y1 = int(np.ceil(max(ty_list))) if ty_list else 0
    y2 = int(np.floor(min([h + t for t in ty_list]))) if ty_list else h

    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))
    return x1, y1, x2, y2


def main():
    ensure_dir(ALIGNED_DIR)
    ensure_dir(BAD_DIR)

    frames = list_frames()

    # 1) Solve first frame for Polaris & build template
    x0, y0, ref_gray = solve_ref_get_polaris_xy(frames[0])
    tmpl = extract_template(ref_gray, (x0, y0), TEMPLATE_HALF)
    H, W = ref_gray.shape  # rows, cols

    # 2) Measure shifts for ALL frames
    shifts = []  # (tx, ty, ok)
    log_rows = []
    for i, p in enumerate(frames, 1):
        g = read_gray_float(p)
        try:
            dx, dy, score, _ = match_shift(g, (x0, y0), tmpl, ROI_HALF)
            tx, ty = -dx, -dy  # translation to APPLY
            ok = bool(score >= MIN_MATCH_SCORE)
        except Exception as e:
            dx = dy = 0.0
            tx = ty = 0.0
            score = 0.0
            ok = False

        shifts.append((tx, ty, ok))
        log_rows.append({
            "frame": p.name,
            "dx_px": round(dx, 3),
            "dy_px": round(dy, 3),
            "apply_tx": round(tx, 3),
            "apply_ty": round(ty, 3),
            "match_score": round(score, 4),
            "ok": int(ok),
        })

        if i % 25 == 0 or i == len(frames):
            print(f"[{i}/{len(frames)}] {p.name}  shift=({dx:.2f},{dy:.2f})  score={score:.3f}  -> {'OK' if ok else 'LOW'}")

    # 3) Compute common crop from ONLY good frames
    good = [(tx, ty) for (tx, ty, ok) in shifts if ok]
    if not good:
        raise RuntimeError("No frames passed the alignment threshold. Lower MIN_MATCH_SCORE or increase ROI_HALF.")

    txs, tys = zip(*good)
    x1, y1, x2, y2 = compute_common_crop(W, H, txs, tys)
    crop_w, crop_h = x2 - x1, y2 - y1
    print(f"Cropping aligned outputs to [{x1}:{x2}, {y1}:{y2}] → {crop_w}×{crop_h}")
    if crop_w < MIN_CROP_WH or crop_h < MIN_CROP_WH:
        raise RuntimeError(f"Common crop too small ({crop_w}×{crop_h}). Too many bad frames or wrong shifts.")

    # 4) Apply translation to COLOR and save 16-bit PNGs for good frames
    saved = 0
    for (p, (tx, ty, ok)) in zip(frames, shifts):
        if not ok:
            # keep a copy of original in BAD_DIR for inspection
            shutil.copy2(p, Path(BAD_DIR) / p.name)
            continue

        rgb16 = read_color16(p)  # H×W×3, uint16, RGB
        M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        warped = cv2.warpAffine(rgb16, M, (W, H), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cropped = warped[y1:y2, x1:x2]  # still RGB uint16

        # Write as true 16-bit PNG via OpenCV; convert RGB→BGR
        bgr16 = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
        out_path = Path(ALIGNED_DIR) / f"{p.stem}_aligned.png"
        cv2.imwrite(str(out_path), bgr16)
        saved += 1

    # 5) Write log CSV
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "dx_px", "dy_px", "apply_tx", "apply_ty", "match_score", "ok"])
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"Done. Saved {saved} aligned frames to {ALIGNED_DIR}. Low-score originals copied to {BAD_DIR}.")
    print(f"Log → {LOG_PATH}")


if __name__ == "__main__":
    main()
