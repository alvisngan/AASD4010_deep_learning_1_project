#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List

import pandas as pd

try:
    import yt_dlp  # type: ignore
except Exception:
    yt_dlp = None

TIME_RE = re.compile(r'^(?:(\d{1,2}):)?(\d{1,2}):(\d{2})$')

# ---------- Utilities ----------

def slugify(text: str, maxlen: int = 70) -> str:
    text = re.sub(r'[^\w\s-]', '', str(text), flags=re.UNICODE)
    text = re.sub(r'[\s_-]+', '_', text).strip('_')
    return text[:maxlen] if len(text) > maxlen else text

def clean_youtube_url(url: str) -> str:
    """Keep only the video id (v=), drop playlist/index params."""
    try:
        u = urllib.parse.urlparse(url)
        qs = urllib.parse.parse_qs(u.query)
        vid = qs.get('v', [None])[0]
        if not vid:
            return url
        return f"https://www.youtube.com/watch?v={vid}"
    except Exception:
        return url

# ---------- Time parsing ----------

def get_video_duration(link: str) -> Optional[float]:
    if yt_dlp is None:
        return None
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'noprogress': True, 'noplaylist': True}) as ydl:
            info = ydl.extract_info(link, download=False)
        return float(info.get('duration')) if info and info.get('duration') is not None else None
    except Exception:
        return None

def parse_hhmmss(s: str, video_duration_s: Optional[float] = None) -> float:
    """
    Parse times robustly:
      - 'MM:SS' -> minutes:seconds
      - 'HH:MM:SS' -> hours:minutes:seconds
      - If video_duration_s < 3600 AND format 'AA:BB:00' with AA,BB<60, treat as MM:SS
      - Accepts tails like '0 days 00:06:00' by taking the trailing HH:MM:SS
    """
    s = str(s).strip()
    tail = s.split()[-1] if (' ' in s and ':' in s) else s
    parts = tail.split(':')
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        m, sec = int(parts[0]), int(parts[1])
        return float(m*60 + sec)
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        a, b, c = int(parts[0]), int(parts[1]), int(parts[2])
        if video_duration_s is not None and video_duration_s < 3600 and c == 0 and 0 <= a < 60 and 0 <= b < 60:
            # Spreadsheet exported 'MM:SS' as 'MM:SS:00' -> interpret as MM:SS
            return float(a*60 + b)
        return float(a*3600 + b*60 + c)
    m = TIME_RE.match(tail)
    if not m:
        raise ValueError(f"Invalid time format: {s!r}")
    hh = int(m.group(1)) if m.group(1) else 0
    mm = int(m.group(2)); ss = int(m.group(3))
    return float(hh*3600 + mm*60 + ss)

# ---------- Data structures ----------

@dataclass
class Segment:
    start_s: float
    end_s: float
    index: int

@dataclass
class SheetMeta:
    cylinders: int
    title: str
    link: str

# ---------- Sheet parsing ----------

def load_sheet(df: pd.DataFrame) -> Tuple[SheetMeta, List[Tuple[str, str]]]:
    required = ['start', 'end', '# of cylinders', 'Title', 'Link']
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing column: {col!r}")
    meta_row_idx = None
    for idx, row in df.iterrows():
        if (not pd.isna(row.get('# of cylinders')) and
            not pd.isna(row.get('Title')) and
            not pd.isna(row.get('Link'))):
            meta_row_idx = idx; break
    if meta_row_idx is None:
        raise KeyError("Could not find metadata row with '# of cylinders', 'Title', and 'Link'")
    meta = SheetMeta(
        int(df.at[meta_row_idx, '# of cylinders']),
        str(df.at[meta_row_idx, 'Title']).strip(),
        clean_youtube_url(str(df.at[meta_row_idx, 'Link']).strip())
    )
    raw_segments: List[Tuple[str, str]] = []
    for _, row in df.iterrows():
        s, e = row.get('start'), row.get('end')
        if pd.isna(s) or pd.isna(e): continue
        raw_segments.append((str(s), str(e)))
    return meta, raw_segments

# ---------- Download ----------

def download_segment_wav(link: str, out_dir: Path, base_name: str, seg: Segment, overwrite: bool=False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base_name}_{seg.index}.wav"
    if out_path.exists() and not overwrite:
        print(f"SKIP (exists): {out_path}"); return out_path
    if yt_dlp is None:
        raise RuntimeError("yt-dlp not installed. Run: pip install yt-dlp")

    # Use yt-dlp to fetch ONLY the range, then extract to WAV directly.
    # Keep quality: 48 kHz, stereo by setting postprocessor args (-ar 48000, -ac 2).
    ranges = [{'start_time': seg.start_s, 'end_time': seg.end_s}]
    def _ranges_cb(info_dict=None, ydl=None):
        return ranges

    ydl_opts = {
        'paths': {'home': str(out_dir)},
        'outtmpl': {'default': f'{base_name}_{seg.index}.%(ext)s'},
        'format': 'bestaudio/best',
        'noplaylist': True,
        'extract_flat': 'discard_in_playlist',
        'force_keyframes_at_cuts': True,
        'download_ranges': _ranges_cb,
        'overwrites': True,
        'quiet': False,
        'noprogress': False,
        'no_warnings': False,
        # Extract to WAV directly
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '0',
        }],
        'postprocessor_args': {
            'FFmpegExtractAudio': ['-ar','48000','-ac','2']
        },
    }

    print(f"+ yt-dlp {seg.index}: {link}  [{seg.start_s:.3f}s -> {seg.end_s:.3f}s]")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])

    # Ensure WAV exists
    if not out_path.exists():
        # Look for any produced wav and rename
        candidates = list(out_dir.glob(f"{base_name}_{seg.index}*.wav"))
        if candidates:
            candidates[0].rename(out_path)
    if not out_path.exists():
        raise FileNotFoundError(f"Segment WAV not found for {base_name}_{seg.index}")
    return out_path

# ---------- Main ----------

def process_workbook(xlsx: Path, out_root: Path, overwrite: bool=False) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    xls = pd.ExcelFile(str(xlsx))
    for sheet in xls.sheet_names:
        print(f"\n=== Sheet: {sheet} ===")
        df = xls.parse(sheet)
        try:
            meta, raw_segments = load_sheet(df)
        except Exception as e:
            print(f"Skipping sheet {sheet!r}: {e}"); continue
        if not raw_segments:
            print("No segments; skipping."); continue

        vid_dur = get_video_duration(meta.link)
        if vid_dur is not None:
            print(f"   (video duration ≈ {vid_dur:.2f}s)")

        # Parse with duration-aware rules
        segs: List[Segment] = []
        idx = 1
        for s_raw, e_raw in raw_segments:
            try:
                s_sec = parse_hhmmss(s_raw, vid_dur)
                e_sec = parse_hhmmss(e_raw, vid_dur)
            except ValueError:
                continue
            if e_sec <= s_sec: continue
            segs.append(Segment(s_sec, e_sec, idx)); idx += 1

        if not segs:
            print("No valid segments after parsing; skipping."); continue

        cyl_dir = out_root / str(meta.cylinders)
        base = slugify(meta.title)
        for seg in segs:
            try:
                download_segment_wav(meta.link, cyl_dir, base, seg, overwrite=overwrite)
            except Exception as e:
                print(f"FAILED segment {seg.index} ({seg.start_s}-{seg.end_s}s): {e}")

def main() -> None:
    ap = argparse.ArgumentParser(description='Download WAV segments via yt-dlp only (sections + extractaudio)')
    ap.add_argument('--xlsx', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    process_workbook(args.xlsx, args.out, overwrite=args.overwrite)

if __name__ == '__main__':
    main()
