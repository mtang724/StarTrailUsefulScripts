# save as make_small_gif.py
from PIL import Image, ImageOps
import glob, os, subprocess, shutil

# ========= TUNE THESE =========
PATTERN = r"cumulative/*.jpg"  # your input sequence
OUT_GIF = "startrail_small.gif"   # compressed GIF path
FPS = 8                       # lower fps -> smaller file (e.g., 8–15)
STEP = 2                       # keep every Nth frame (2 halves frames; 1 keeps all)
MAX_WIDTH = 720               # scale down if wider than this (keep aspect)
PALETTE_COLORS = 64           # 64/128/256; fewer colors -> smaller file
DITHER = Image.Dither.FLOYDSTEINBERG  # or Image.Dither.NONE for cleaner, smaller
# ==============================

def load_paths():
    files = sorted(glob.glob(PATTERN))
    if not files:
        raise SystemExit(f"No files matched pattern: {PATTERN}")
    # Drop frames (subsample)
    return files[::max(1, STEP)]

def open_and_prepare(path, target_w):
    im = Image.open(path).convert("RGB")
    # Resize if needed (preserve aspect)
    if im.width > target_w:
        new_h = int(im.height * (target_w / im.width))
        im = im.resize((target_w, new_h), Image.Resampling.LANCZOS)
    # Quantize to limited palette for big savings
    im = im.quantize(colors=PALETTE_COLORS, method=Image.MEDIANCUT, dither=DITHER)
    # Ensure disposal=2 works cleanly by forcing full frames (avoid partial updates)
    im = ImageOps.contain(im, im.size)  # no-op but keeps mode/size consistent
    return im

def make_gif():
    files = load_paths()
    # Determine base width from first frame (after optional cap)
    with Image.open(files[0]) as f0:
        base_w = min(f0.width, MAX_WIDTH)

    frames = [open_and_prepare(p, base_w) for p in files]
    if len(frames) == 1:
        # single-frame GIF still okay
        duration_ms = int(1000 / max(FPS, 1))
        frames[0].save(OUT_GIF, save_all=True, duration=duration_ms, loop=0, optimize=True, disposal=2)
    else:
        duration_ms = int(1000 / max(FPS, 1))
        frames[0].save(
            OUT_GIF,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
            disposal=2,
        )
    print(f"Saved GIF: {OUT_GIF} ({os.path.getsize(OUT_GIF)/1024/1024:.2f} MB)")
    return OUT_GIF

def maybe_convert_to_mp4(gif_path):
    """Optional: convert the GIF to a much smaller MP4 (H.264) if ffmpeg exists."""
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found in PATH — skipping MP4 conversion.")
        return
    mp4_path = os.path.splitext(gif_path)[0] + ".mp4"
    # -movflags +faststart for web, -pix_fmt yuv420p for compatibility
    cmd = [
        "ffmpeg", "-y", "-i", gif_path,
        "-movflags", "+faststart", "-pix_fmt", "yuv420p",
        "-vf", f"fps={FPS}",
        "-c:v", "libx264", mp4_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Also saved MP4: {mp4_path} ({os.path.getsize(mp4_path)/1024/1024:.2f} MB)")
    except subprocess.CalledProcessError as e:
        print("ffmpeg failed to convert GIF to MP4.")
        print(e.stderr.decode(errors="ignore")[:500])

if __name__ == "__main__":
    gif = make_gif()
    # maybe_convert_to_mp4(gif)
