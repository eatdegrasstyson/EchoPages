import os, sys, re
from pathlib import Path
import yt_dlp
from SpotifyToSpectrogram.get_metadata import get_data_from_id

# ffmpeg locations
#FFMPEG_DIR   = Path(sys.prefix) / "Library" / "bin"
FFMPEG_DIR   = Path(r"C:\FFMPEG\bin")
FFMPEG_EXE   = FFMPEG_DIR / "ffmpeg.EXE"
FFPROBE_EXE  = FFMPEG_DIR / "ffprobe.EXE"

# Define audio output path using pathlib

def safe(name: str) -> str:
    """Sanitize filenames for OS safety."""
    return re.sub(r'[\\/*?:"<>|]', "_", name)

def download_mp3_from_spotify_id(track, output_dir):
    """Download an MP3 from a Spotify track description."""
    artist, title, dur_s = track[0], track[1], track[2] / 1000
    print(artist, title, dur_s)

    query = f"{artist} - {title} official audio"
    # lo, hi = max(1, dur_s - 20), dur_s + 20

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / f"{safe(artist)} - {safe(title)}.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        # "match_filter": yt_dlp.utils.match_filter_func(f"duration >= {lo} & duration <= {hi}"),
        "ffmpeg_location": str(FFMPEG_EXE),
        "prefer_ffmpeg": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "postprocessor_args": ["-c:a", "mp3_mf", "-b:a", "192k"],
        "retries": 3,
        "fragment_retries": 3,
        "extractor_retries": 3,
        "forceip": "0",
        "extractor_args": {"youtube": {"player_client": ["android"]}},
    }

    last_err = None
    for prefix in ("ytmusicsearch1:", "ytsearch1:"):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(prefix + query, download=True)
                entry = info["entries"][0] if "entries" in info else info
                filepath = ydl.prepare_filename(entry)
                mp3_path = Path(filepath).with_suffix(".mp3")
                return mp3_path
        except Exception as e:
            last_err = str(e)
            print('=' * len(last_err) + '\n' + last_err + '\n' + '=' * len(last_err))

    # raise RuntimeError(f"Failed to find/download audio: {last_err}")
