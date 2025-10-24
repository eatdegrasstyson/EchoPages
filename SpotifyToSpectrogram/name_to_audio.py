import os, sys, re, shutil
import yt_dlp
from SpotifyToSpectrogram.get_metadata import get_data_from_id

FFMPEG_DIR   = os.path.join(sys.prefix, "Library", "bin")
FFMPEG_EXE   = os.path.join(FFMPEG_DIR, "ffmpeg.EXE")
FFPROBE_EXE  = os.path.join(FFMPEG_DIR, "ffprobe.EXE")
DOWNLOAD_DIR = r"C:\Users\jaina\Dropbox\EchoPages\DataSets\audio"


def safe(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', '_', name)

def download_mp3_from_spotify_id(track):
    artist, title, dur_s = track[0], track[1], track[2] / 1000
    print(artist, title, dur_s)

    query = f'{artist} - {title} official audio'
    lo, hi = max(1, dur_s - 7), dur_s + 7

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{DOWNLOAD_DIR}/{safe(artist)} - {safe(title)}.%(ext)s",
        "noplaylist": True,
        "quiet": True,
        "match_filter": yt_dlp.utils.match_filter_func(f"duration >= {lo} & duration <= {hi}"),
        # 5) Give yt-dlp the explicit *file* path (not just the dir)
        "ffmpeg_location": FFMPEG_EXE,
        # Optional: be explicit about using ffmpeg
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
                filename = ydl.prepare_filename(entry).rsplit('.', 1)[0]
                return filename + ".mp3"
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to find/download audio: {last_err}")
