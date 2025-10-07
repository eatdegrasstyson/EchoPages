import re, yt_dlp
from id_to_name import get_data_from_id

def safe(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', '_', name)

def download_mp3_from_spotify_id(track):

    artist, title, dur_s = track['artists'][0]['name'], \
                           track['name'], track['duration_ms']*1000
    
    print(artist, title, dur_s)
    
    query = f'{artist} - {title} official audio'
    # duration window ±7s to avoid live/remix; tweak as needed
    lo, hi = max(1, dur_s - 7), dur_s + 7

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{safe(artist)} - {safe(title)}.%(ext)s",
        "noplaylist": True,
        "quiet": True,
        # prefer results near the Spotify duration
        "match_filter": yt_dlp.utils.match_filter_func(f"duration >= {lo} & duration <= {hi}"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }

    # Try YouTube Music search first, then regular YouTube
    for prefix in ( "ytmusicsearch1:", "ytsearch1:" ):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(prefix + query, download=True)
                # when using search, extract_info returns a 'entries' list
                entry = info["entries"][0] if "entries" in info else info
                return ydl.prepare_filename(entry).rsplit('.', 1)[0] + ".mp3"
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to find/download audio: {last_err}")

# usage:
track = get_data_from_id('3n3Ppam7vgaVa1iaRUc9Lp')
path = download_mp3_from_spotify_id(track)
print("Saved:", path)
