#!/usr/bin/env python3
"""
contact_sheets.py
─────────────────
• Videos → 9-frame sheet with timestamp
• Pictures → 40-thumb sheets
• Real-time log, per-folder counters, skip list, --shallow
• Optional modified-date filtering (before / after cutoff)
• Desktop GUI for configuring and running jobs
"""

import argparse
import hashlib
import math
import os
import queue
import subprocess
import tempfile
import threading
import time
import unicodedata
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MAX_BASENAME = 240

# ─── USER SETTINGS ──────────────────────────────────────────────────────────
VIDEO_THUMBS = 9
VIDEO_GRID_COLS = 3
PICS_PER_SHEET = 40
PIC_GRID_COLS = 8
FRAME_WIDTH = 640
JPEG_QUALITY = 75
CHECK_EVERY = 50
SKIP_FOLDERS = {
    # "/Volumes/Pesx/Archive",
    # "System Files",
}
VIDEO_EXTS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".flv",
    ".wmv",
    ".mpg",
    ".mpeg",
    ".m4v",
    ".3gp",
    ".ts",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
DEFAULT_DEST_ROOT = Path.home() / "Downloads" / "contact_sheets"
IGNORED_PREFIXES = {"._"}
MIN_BYTES = 32 * 1024
MAX_WORKERS = os.cpu_count() or 4
TIMESTAMP_PCTS = [0.05, 0.15, 0.25, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
]
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class RunOptions:
    roots: list[Path]
    recursive: bool = True
    dest_root: Path = DEFAULT_DEST_ROOT
    date_mode: str = "any"  # any | before | after
    cutoff: datetime | None = None
    workers: int = MAX_WORKERS


def ts() -> str:
    return datetime.now().strftime("%Y.%m.%d %I:%M:%S%p").lower()


def safe_stem(stem: str) -> str:
    stem = unicodedata.normalize("NFC", stem)
    if len(stem.encode()) <= MAX_BASENAME:
        return stem
    h = hashlib.sha1(stem.encode()).hexdigest()[:8]
    trunc = stem.encode()[: MAX_BASENAME - 9].decode(errors="ignore")
    return f"{trunc}_{h}"


def safe_save(sheet: Image.Image, dest: Path) -> bool:
    try:
        sheet.save(dest, "JPEG", quality=JPEG_QUALITY, optimize=True)
        return True
    except OSError as e:
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


def mirror_path(src_root: Path, p: Path) -> Path:
    if p.is_absolute() and p.parts[:2] == ("/", "Volumes"):
        return p.relative_to(Path("/Volumes"))
    return Path(src_root.name) / p.relative_to(src_root)


def should_skip_folder(path: Path) -> bool:
    spath = str(path)
    for skip in SKIP_FOLDERS:
        if skip.startswith("/"):
            if spath.startswith(skip):
                return True
        elif skip in path.parts:
            return True
    return False


def parse_date_input(raw: str) -> datetime:
    text = raw.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(
        "Invalid date format. Use YYYY-MM-DD (optionally with HH:MM or HH:MM:SS)."
    )


def within_date_filter(st_mtime: float, mode: str, cutoff: datetime | None) -> bool:
    if mode == "any" or cutoff is None:
        return True
    file_dt = datetime.fromtimestamp(st_mtime)
    if mode == "before":
        return file_dt <= cutoff
    if mode == "after":
        return file_dt >= cutoff
    return True


def probe_duration(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    ).strip()
    return float(out) if out else 0.0


def extract_frame(clip: Path, t: float, out_file: Path):
    label = time.strftime("%H:%M:%S", time.gmtime(t)).replace(":", r"\:")
    vf = (
        f"scale={FRAME_WIDTH}:-1,"
        f"drawtext=text='{label}':x=w-tw-4:y=h-th-4:"
        f"fontsize=18:fontcolor=white:borderw=2:bordercolor=black"
    )
    subprocess.check_call(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{t:.3f}",
            "-i",
            str(clip),
            "-an",
            "-frames:v",
            "1",
            "-vf",
            vf,
            str(out_file),
        ],
        stderr=subprocess.DEVNULL,
    )


def make_varsheet(thumbs, cols, dest: Path) -> bool:
    rows = math.ceil(len(thumbs) / cols)
    w = FRAME_WIDTH
    row_h = [max(im.size[1] for im in thumbs[r * cols : (r + 1) * cols]) for r in range(rows)]
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


def process_video(clip: Path, root: Path, dest_root: Path):
    folder = clip.parent
    dest_dir = dest_root / mirror_path(root, folder)
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
                extract_frame(clip, pct * dur, f)
                thumbs.append(Image.open(f))
            ok = make_varsheet(thumbs, VIDEO_GRID_COLS, out_file)
        return folder, not ok, f"✔ {clip}" if ok else f"⚠ {clip} (save fail)"
    except Exception as e:
        return folder, False, f"⚠ {clip} — {e}"


def process_photo_folder(folder: Path, imgs, root: Path, dest_root: Path):
    dest_dir = dest_root / mirror_path(root, folder)
    dest_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(imgs, key=lambda p: p.name.lower())
    batches = [imgs[i : i + PICS_PER_SHEET] for i in range(0, len(imgs), PICS_PER_SHEET)]

    msgs, any_ok = [], False
    for idx, batch in enumerate(batches, 1):
        out_file = dest_dir / f"photos_contact_{idx:02d}.jpg"
        if out_file.exists():
            continue
        thumbs = []
        for pic in batch:
            try:
                im = Image.open(pic)
                im.thumbnail((FRAME_WIDTH, FRAME_WIDTH * 10_000), Image.LANCZOS)
                thumbs.append(im)
            except Exception:
                continue
        if not thumbs:
            continue
        if make_varsheet(thumbs, PIC_GRID_COLS, out_file):
            any_ok = True
            msgs.append(f"✔ {out_file}")
    return folder, not any_ok, " | ".join(msgs)


def collect_root_tasks(options: RunOptions, root: Path):
    videos, pic_folders = [], {}
    folder_tot = defaultdict(int)
    it = root.rglob("*") if options.recursive else root.iterdir()
    for p in it:
        if should_skip_folder(p.parent):
            continue
        try:
            st = p.stat()
        except FileNotFoundError:
            continue
        if st.st_size < MIN_BYTES or p.name.startswith(tuple(IGNORED_PREFIXES)):
            continue
        if not within_date_filter(st.st_mtime, options.date_mode, options.cutoff):
            continue
        ext = p.suffix.lower()
        if ext in VIDEO_EXTS:
            videos.append(p)
            folder_tot[p.parent] += 1
        elif ext in IMAGE_EXTS:
            pic_folders.setdefault(p.parent, []).append(p)
    for folder, imgs in pic_folders.items():
        folder_tot[folder] += math.ceil(len(imgs) / PICS_PER_SHEET)
    return videos, pic_folders, folder_tot


def run_jobs(options: RunOptions, log, progress):
    all_vids, all_pics, folder_totals = [], {}, defaultdict(int)
    for root in options.roots:
        vids, pics, ft = collect_root_tasks(options, root)
        all_vids.extend((v, root) for v in vids)
        for k, lst in pics.items():
            all_pics.setdefault((k, root), []).extend(lst)
        for k, n in ft.items():
            folder_totals[k] += n

    total = len(all_vids) + len(all_pics)
    log(
        f"{ts()} {len(all_vids)} videos • {len(all_pics)} picture folders → "
        f"{total} jobs on {options.workers} cores"
    )

    if total == 0:
        progress(0, 0, 0, 0.0, 0.0)
        log(f"{ts()} Nothing matched your filters.")
        return

    folder_done = defaultdict(int)
    skipped = done = 0
    start = time.time()

    with ProcessPoolExecutor(options.workers) as pool:
        futs = [pool.submit(process_video, v, r, options.dest_root) for v, r in all_vids] + [
            pool.submit(process_photo_folder, f, imgs, r, options.dest_root)
            for (f, r), imgs in all_pics.items()
        ]

        for fut in as_completed(futs):
            folder, was_skip, msg = fut.result()
            done += 1
            folder_done[folder] += 1
            if was_skip:
                skipped += 1
            elif msg:
                log(
                    f"{ts()} {done}/{total}, {folder_done[folder]}/"
                    f"{folder_totals[folder]} {msg}"
                )

            elapsed = time.time() - start
            pct = done * 100 / total
            eta = elapsed / done * (total - done) if done < total else 0
            progress(done, total, skipped, pct, eta)

            if done % CHECK_EVERY == 0 or done == total:
                hms = lambda s: time.strftime("%H:%M:%S", time.gmtime(s))
                log(
                    f"{ts()} PROGRESS {done}/{total} ({pct:.1f}%) • "
                    f"elapsed {hms(elapsed)} • ETA {hms(eta)} • skipped {skipped}"
                )

    log(f"{ts()} Finished • {done} jobs • {skipped} skipped")


def run_cli(args):
    roots = [Path(p).expanduser() for p in args.src]
    options = RunOptions(
        roots=roots,
        recursive=not args.shallow,
        dest_root=Path(args.dest).expanduser(),
        date_mode=args.date_mode,
        cutoff=parse_date_input(args.date) if args.date else None,
        workers=max(1, args.workers),
    )

    run_jobs(
        options,
        log=lambda line: print(line, flush=True),
        progress=lambda *_: None,
    )


def launch_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    class ContactSheetsApp(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Contact Sheets Builder")
            self.geometry("1100x760")
            self.minsize(1000, 700)

            self.task_queue: queue.Queue = queue.Queue()
            self.worker_thread: threading.Thread | None = None

            self.sources: list[Path] = []

            self._build_styles()
            self._build_ui()
            self.after(120, self._poll_queue)

        def _build_styles(self):
            style = ttk.Style(self)
            style.theme_use("clam")
            style.configure("Root.TFrame", background="#111827")
            style.configure("Card.TFrame", background="#1F2937", relief="flat")
            style.configure("TLabel", background="#1F2937", foreground="#E5E7EB")
            style.configure("Title.TLabel", font=("Segoe UI", 22, "bold"))
            style.configure("Sub.TLabel", font=("Segoe UI", 10), foreground="#9CA3AF")
            style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
            style.configure("TCheckbutton", background="#1F2937", foreground="#E5E7EB")
            style.configure("TRadiobutton", background="#1F2937", foreground="#E5E7EB")
            style.configure("TEntry", fieldbackground="#111827", foreground="#E5E7EB")

        def _build_ui(self):
            root = ttk.Frame(self, padding=16, style="Root.TFrame")
            root.pack(fill="both", expand=True)

            card = ttk.Frame(root, padding=16, style="Card.TFrame")
            card.pack(fill="both", expand=True)

            ttk.Label(card, text="Contact Sheets Builder", style="Title.TLabel").pack(anchor="w")
            ttk.Label(
                card,
                text="Create video and photo contact sheets with date filtering.",
                style="Sub.TLabel",
            ).pack(anchor="w", pady=(2, 14))

            src_row = ttk.Frame(card, style="Card.TFrame")
            src_row.pack(fill="x", pady=4)
            ttk.Label(src_row, text="Source folders:").pack(side="left")
            ttk.Button(src_row, text="Add Folder", command=self._add_source).pack(side="left", padx=8)
            ttk.Button(src_row, text="Remove Selected", command=self._remove_source).pack(side="left")
            ttk.Button(src_row, text="Clear", command=self._clear_sources).pack(side="left", padx=8)

            self.src_list = tk.Listbox(
                card,
                height=8,
                bg="#111827",
                fg="#E5E7EB",
                selectbackground="#2563EB",
                relief="flat",
                highlightthickness=0,
            )
            self.src_list.pack(fill="x", pady=(4, 10))

            opts = ttk.Frame(card, style="Card.TFrame")
            opts.pack(fill="x", pady=4)

            self.shallow_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                opts,
                text="Shallow scan (top-level files only)",
                variable=self.shallow_var,
            ).grid(row=0, column=0, sticky="w", padx=(0, 18), pady=3)

            ttk.Label(opts, text="Workers:").grid(row=0, column=1, sticky="e")
            self.workers_var = tk.StringVar(value=str(MAX_WORKERS))
            ttk.Entry(opts, width=6, textvariable=self.workers_var).grid(
                row=0, column=2, sticky="w", padx=6
            )

            ttk.Label(opts, text="Output folder:").grid(row=1, column=0, sticky="w", pady=6)
            self.dest_var = tk.StringVar(value=str(DEFAULT_DEST_ROOT))
            ttk.Entry(opts, textvariable=self.dest_var).grid(row=1, column=1, columnspan=2, sticky="ew")
            ttk.Button(opts, text="Browse", command=self._choose_dest).grid(row=1, column=3, padx=8)

            opts.columnconfigure(1, weight=1)

            date_frame = ttk.LabelFrame(card, text="Modified-date filter", padding=10)
            date_frame.pack(fill="x", pady=10)

            self.date_mode_var = tk.StringVar(value="any")
            ttk.Radiobutton(date_frame, text="All files", value="any", variable=self.date_mode_var).grid(
                row=0, column=0, sticky="w", padx=(0, 18)
            )
            ttk.Radiobutton(
                date_frame,
                text="Only files modified on/before date",
                value="before",
                variable=self.date_mode_var,
            ).grid(row=0, column=1, sticky="w", padx=(0, 18))
            ttk.Radiobutton(
                date_frame,
                text="Only files modified on/after date",
                value="after",
                variable=self.date_mode_var,
            ).grid(row=0, column=2, sticky="w")

            ttk.Label(date_frame, text="Cutoff date/time:").grid(row=1, column=0, sticky="w", pady=(8, 0))
            self.date_var = tk.StringVar()
            ttk.Entry(date_frame, width=28, textvariable=self.date_var).grid(
                row=1, column=1, sticky="w", pady=(8, 0)
            )
            ttk.Label(
                date_frame,
                text="Formats: YYYY-MM-DD or YYYY-MM-DD HH:MM[:SS]",
                style="Sub.TLabel",
            ).grid(row=1, column=2, sticky="w", pady=(8, 0))

            ctl = ttk.Frame(card, style="Card.TFrame")
            ctl.pack(fill="x", pady=(8, 4))

            self.start_btn = ttk.Button(ctl, text="Start Processing", style="Accent.TButton", command=self._start)
            self.start_btn.pack(side="left")

            self.progress_var = tk.StringVar(value="Idle")
            ttk.Label(ctl, textvariable=self.progress_var).pack(side="left", padx=14)

            self.progress = ttk.Progressbar(card, orient="horizontal", mode="determinate")
            self.progress.pack(fill="x", pady=(2, 8))

            self.log_box = tk.Text(
                card,
                wrap="word",
                height=18,
                bg="#0B1220",
                fg="#E5E7EB",
                insertbackground="#E5E7EB",
                relief="flat",
                highlightthickness=0,
            )
            self.log_box.pack(fill="both", expand=True)

        def _add_source(self):
            selected = filedialog.askdirectory(title="Choose source folder")
            if not selected:
                return
            p = Path(selected)
            if p not in self.sources:
                self.sources.append(p)
                self.src_list.insert("end", str(p))

        def _remove_source(self):
            idxs = list(self.src_list.curselection())
            if not idxs:
                return
            for idx in reversed(idxs):
                del self.sources[idx]
                self.src_list.delete(idx)

        def _clear_sources(self):
            self.sources.clear()
            self.src_list.delete(0, "end")

        def _choose_dest(self):
            selected = filedialog.askdirectory(title="Choose output folder")
            if selected:
                self.dest_var.set(selected)

        def _log(self, message: str):
            self.log_box.insert("end", message + "\n")
            self.log_box.see("end")

        def _validate_options(self) -> RunOptions | None:
            if not self.sources:
                messagebox.showerror("Missing source", "Add at least one source folder.")
                return None

            try:
                workers = max(1, int(self.workers_var.get().strip()))
            except ValueError:
                messagebox.showerror("Invalid workers", "Workers must be a positive integer.")
                return None

            mode = self.date_mode_var.get()
            cutoff = None
            if mode in {"before", "after"}:
                raw = self.date_var.get().strip()
                if not raw:
                    messagebox.showerror("Missing date", "Provide a cutoff date for the selected filter.")
                    return None
                try:
                    cutoff = parse_date_input(raw)
                except ValueError as e:
                    messagebox.showerror("Invalid date", str(e))
                    return None

            options = RunOptions(
                roots=[Path(p).expanduser() for p in self.sources],
                recursive=not self.shallow_var.get(),
                dest_root=Path(self.dest_var.get().strip() or str(DEFAULT_DEST_ROOT)).expanduser(),
                date_mode=mode,
                cutoff=cutoff,
                workers=workers,
            )
            return options

        def _set_running(self, running: bool):
            state = "disabled" if running else "normal"
            self.start_btn.configure(state=state)

        def _start(self):
            options = self._validate_options()
            if not options:
                return
            self._set_running(True)
            self.progress.configure(value=0, maximum=100)
            self.progress_var.set("Running...")
            self._log("=" * 72)
            self._log(f"{ts()} Starting run")

            def worker():
                try:
                    run_jobs(
                        options,
                        log=lambda m: self.task_queue.put(("log", m)),
                        progress=lambda d, t, s, p, e: self.task_queue.put(
                            ("progress", (d, t, s, p, e))
                        ),
                    )
                    self.task_queue.put(("done", None))
                except Exception as e:
                    self.task_queue.put(("error", str(e)))

            self.worker_thread = threading.Thread(target=worker, daemon=True)
            self.worker_thread.start()

        def _poll_queue(self):
            try:
                while True:
                    kind, payload = self.task_queue.get_nowait()
                    if kind == "log":
                        self._log(payload)
                    elif kind == "progress":
                        done, total, skipped, pct, eta = payload
                        self.progress.configure(value=pct)
                        eta_hms = time.strftime("%H:%M:%S", time.gmtime(eta))
                        self.progress_var.set(
                            f"{done}/{total} ({pct:.1f}%) • skipped {skipped} • ETA {eta_hms}"
                        )
                    elif kind == "done":
                        self.progress_var.set("Completed")
                        self._set_running(False)
                    elif kind == "error":
                        self._log(f"{ts()} ERROR: {payload}")
                        self.progress_var.set("Failed")
                        self._set_running(False)
                        messagebox.showerror("Processing error", payload)
            except queue.Empty:
                pass
            self.after(120, self._poll_queue)


    app = ContactSheetsApp()
    app.mainloop()

def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="*", help="one or more source folders/drives")
    ap.add_argument("--shallow", action="store_true", help="process only top-level files")
    ap.add_argument(
        "--dest",
        default=str(DEFAULT_DEST_ROOT),
        help="destination root for contact sheets",
    )
    ap.add_argument(
        "--date-mode",
        choices=["any", "before", "after"],
        default="any",
        help="filter by modified date",
    )
    ap.add_argument(
        "--date",
        help="cutoff date (YYYY-MM-DD, optionally HH:MM or HH:MM:SS)",
    )
    ap.add_argument("--workers", type=int, default=MAX_WORKERS, help="parallel workers")
    ap.add_argument(
        "--cli",
        action="store_true",
        help="run in CLI mode (default launches GUI unless src is provided)",
    )
    return ap


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cli or args.src:
        if not args.src:
            parser.error("CLI mode requires at least one src path")
        if args.date_mode in {"before", "after"} and not args.date:
            parser.error("--date is required when --date-mode is before or after")
        if args.date:
            try:
                parse_date_input(args.date)
            except ValueError as exc:
                parser.error(str(exc))
        run_cli(args)
        return

    launch_gui()


if __name__ == "__main__":
    main()
