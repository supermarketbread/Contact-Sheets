#!/usr/bin/env python3
"""
fast_contact_sheets.py  –  resilient version
──────────────────────
• Videos → 9-frame sheet with timestamp
• Pictures → 40-thumb sheets
• Real-time log, per-folder counters, skip list, --shallow
• Protects against huge filenames, huge images, broken data streams
"""

import argparse, subprocess, os, sys, math, tempfile, time, hashlib, unicodedata
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from collections import defaultdict
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None          # no decompression-bomb limit
ImageFile.LOAD_TRUNCATED_IMAGES = True # tolerate truncated JPEGs
MAX_BASENAME = 240                     # 240 + suffix < 255 bytes

# ─── USER SETTINGS ──────────────────────────────────────────────────────────
VIDEO_THUMBS    = 9
VIDEO_GRID_COLS = 3
PICS_PER_SHEET  = 40
PIC_GRID_COLS   = 8
FRAME_WIDTH     = 640
JPEG_QUALITY    = 75
CHECK_EVERY     = 50
SKIP_FOLDERS = {
    # "/Volumes/Pesx/Archive",
    # "System Files",
}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm",
              ".flv", ".wmv", ".mpg", ".mpeg", ".m4v", ".3gp", ".ts"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
DEST_ROOT   = Path.home() / "Downloads" / "contact_sheets"
IGNORED_PREFIXES = {"._"}
MIN_BYTES   = 32 * 1024
MAX_WORKERS = os.cpu_count() or 4
TIMESTAMP_PCTS = [0.05, 0.15, 0.25, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
# ────────────────────────────────────────────────────────────────────────────


def ts() -> str:
    return datetime.now().strftime("%Y.%m.%d %I:%M:%S%p").lower()


# ─── helper: safe filename ──────────────────────────────────────────────────
def safe_stem(stem: str) -> str:
    stem = unicodedata.normalize("NFC", stem)
    if len(stem.encode()) <= MAX_BASENAME:
        return stem
    h = hashlib.sha1(stem.encode()).hexdigest()[:8]
    trunc = stem.encode()[:MAX_BASENAME-9].decode(errors="ignore")
    return f"{trunc}_{h}"


# ─── helper: save, shrinking tall sheets if needed ──────────────────────────
def safe_save(sheet: Image.Image, dest: Path) -> bool:
    try:
        sheet.save(dest, "JPEG", quality=JPEG_QUALITY, optimize=True)
        return True
    except OSError as e:
        # Hit 65 500-pixel limit or corrupt data – try shrinking, else skip
        if "65500" in str(e) or "height or width exceeds limit" in str(e):
            ratio = 65500 / sheet.height
            w = int(sheet.width * ratio)
            h = 65500
            sheet = sheet.resize((w, h), Image.LANCZOS)
            try:
                sheet.save(dest, "JPEG", quality=JPEG_QUALITY, optimize=True)
                return True
            except Exception:
                return False
        return False


# ─── path helpers ───────────────────────────────────────────────────────────
def mirror_path(src_root: Path, p: Path) -> Path:
    if p.is_absolute() and p.parts[:2] == ("/", "Volumes"):
        return p.relative_to(Path("/Volumes"))
    return Path(src_root.name) / p.relative_to(src_root)


def should_skip_folder(path: Path) -> bool:
    spath = str(path)
    for skip in SKIP_FOLDERS:
        if skip.startswith('/'):
            if spath.startswith(skip):
                return True
        elif skip in path.parts:
            return True
    return False


# ─── ffmpeg helpers ─────────────────────────────────────────────────────────
def probe_duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        text=True).strip()
    return float(out) if out else 0.0


def extract_frame(clip: Path, t: float, out_file: Path):
    label = time.strftime("%H:%M:%S", time.gmtime(t)).replace(":", r"\:")
    vf = (f"scale={FRAME_WIDTH}:-1,"
          f"drawtext=text='{label}':x=w-tw-4:y=h-th-4:"
          f"fontsize=18:fontcolor=white:borderw=2:bordercolor=black")
    subprocess.check_call(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-ss", f"{t:.3f}", "-i", str(clip),
         "-an", "-frames:v", "1", "-vf", vf, str(out_file)],
        stderr=subprocess.DEVNULL
    )


def make_varsheet(thumbs, cols, dest: Path) -> bool:
    rows = math.ceil(len(thumbs) / cols)
    w = FRAME_WIDTH
    row_h = [max(im.size[1] for im in thumbs[r*cols:(r+1)*cols]) for r in range(rows)]
    sheet = Image.new("RGB", (w * cols, sum(row_h)), "black")

    y = idx = 0
    for h in row_h:
        x = 0
        for _ in range(cols):
            if idx >= len(thumbs):
                break
            sheet.paste(thumbs[idx], (x, y))
            x += w
            idx += 1
        y += h
    return safe_save(sheet, dest)


# ─── worker tasks ───────────────────────────────────────────────────────────
def process_video(clip: Path, root: Path):
    folder = clip.parent
    dest_dir = DEST_ROOT / mirror_path(root, folder)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_file = dest_dir / f"{safe_stem(clip.stem)}_contact.jpg"
    if out_file.exists():
        return folder, True, ""
    try:
        dur = probe_duration(clip) or 1.0
        with tempfile.TemporaryDirectory() as td:
            thumbs = []
            for pct in TIMESTAMP_PCTS:
                f = Path(td) / f"{pct:.2f}.jpg"
                extract_frame(clip, pct*dur, f)
                thumbs.append(Image.open(f))
            ok = make_varsheet(thumbs, VIDEO_GRID_COLS, out_file)
        return folder, not ok, f"✔ {clip}" if ok else f"⚠ {clip} (save fail)"
    except Exception as e:
        return folder, False, f"⚠ {clip} — {e}"


def process_photo_folder(folder: Path, imgs, root: Path):
    dest_dir = DEST_ROOT / mirror_path(root, folder)
    dest_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(imgs, key=lambda p: p.name.lower())
    batches = [imgs[i:i+PICS_PER_SHEET] for i in range(0, len(imgs), PICS_PER_SHEET)]

    msgs, any_ok = [], False
    for idx, batch in enumerate(batches, 1):
        out_file = dest_dir / f"photos_contact_{idx:02d}.jpg"
        if out_file.exists(): continue
        thumbs = []
        for pic in batch:
            try:
                im = Image.open(pic)
                im.thumbnail((FRAME_WIDTH, FRAME_WIDTH*10_000), Image.LANCZOS)
                thumbs.append(im)
            except Exception:
                continue
        if not thumbs:
            continue
        if make_varsheet(thumbs, PIC_GRID_COLS, out_file):
            any_ok = True
            msgs.append(f"✔ {out_file}")
    return folder, not any_ok, " | ".join(msgs)


# ─── task collection per root ───────────────────────────────────────────────
def collect_root_tasks(root: Path, recursive: bool):
    videos, pic_folders = [], {}
    folder_tot = defaultdict(int)
    it = root.rglob('*') if recursive else root.iterdir()
    for p in it:
        if should_skip_folder(p.parent): continue
        try:
            st = p.stat()
        except FileNotFoundError:
            continue
        if st.st_size < MIN_BYTES or p.name.startswith(tuple(IGNORED_PREFIXES)):
            continue
        ext = p.suffix.lower()
        if ext in VIDEO_EXTS:
            videos.append(p); folder_tot[p.parent] += 1
        elif ext in IMAGE_EXTS:
            pic_folders.setdefault(p.parent, []).append(p)
    for folder, imgs in pic_folders.items():
        folder_tot[folder] += math.ceil(len(imgs) / PICS_PER_SHEET)
    return videos, pic_folders, folder_tot


# ─── main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs='+', help="one or more source folders/drives")
    ap.add_argument("--shallow", action="store_true",
                    help="process only top-level files (no recursion)")
    args = ap.parse_args()

    roots = [Path(p).expanduser() for p in args.src]
    recursive = not args.shallow

    all_vids, all_pics, folder_totals = [], {}, defaultdict(int)
    for root in roots:
        vids, pics, ft = collect_root_tasks(root, recursive)
        all_vids.extend((v, root) for v in vids)
        for k, lst in pics.items():
            all_pics.setdefault((k, root), []).extend(lst)
        for k, n in ft.items():
            folder_totals[k] += n

    total = len(all_vids) + len(all_pics)
    print(f"{ts()} {len(all_vids)} videos • {len(all_pics)} picture folders → "
          f"{total} jobs on {MAX_WORKERS} cores\n")

    folder_done = defaultdict(int); skipped = done = 0; start = time.time()

    with ProcessPoolExecutor(MAX_WORKERS) as pool:
        futs = ([pool.submit(process_video, v, r) for v, r in all_vids] +
                [pool.submit(process_photo_folder, f, imgs, r)
                 for (f, r), imgs in all_pics.items()])

        for fut in as_completed(futs):
            folder, was_skip, msg = fut.result()
            done += 1; folder_done[folder] += 1
            if was_skip: skipped += 1
            else:        print(f"{ts()} {done}/{total}, {folder_done[folder]}/"
                                f"{folder_totals[folder]} {msg}", flush=True)

            if done % CHECK_EVERY == 0 or done == total:
                el = time.time() - start
                pct = done*100/total
                eta = el/done*(total-done) if done < total else 0
                hms = lambda s: time.strftime('%H:%M:%S', time.gmtime(s))
                print(f"\033[96m{ts()} PROGRESS {done}/{total} ({pct:.1f}%) • "
                      f"elapsed {hms(el)} • ETA {hms(eta)} • skipped {skipped}\033[0m",
                      flush=True)

    print(f"{ts()} Finished • {done} jobs • {skipped} skipped")


if __name__ == "__main__":
    main()
