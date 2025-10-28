import pandas as pd
from pathlib import Path
from SpotifyToSpectrogram.get_metadata import get_data_from_id
from SpotifyToSpectrogram.name_to_audio import download_mp3_from_spotify_id
from SpotifyToSpectrogram.mp3ToSpectrogram import plot_spectrogram_and_save

IMG_OUTPUT_PATH = Path("DataSets/img")
AUDIO_OUTPUT_PATH = Path("DataSets/audio")

IMG_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
AUDIO_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


df = pd.read_csv("DataSets/SpotifyMetaToGems_Final.csv")

rows = []  # collect rows, then build a DataFrame once

for spotify_id in df["spotifyid"]:
    track = get_data_from_id(spotify_id)
    full_audio_path = download_mp3_from_spotify_id(track, AUDIO_OUTPUT_PATH)

    file_name = Path(full_audio_path).stem
    audio_path = AUDIO_OUTPUT_PATH / f"{file_name}"
    spectro_path = IMG_OUTPUT_PATH / f"{file_name}"

    # create and save spectrogram
    plot_spectrogram_and_save(full_audio_path, spectro_path)

    # append one new row
    rows.append({
        "spotifyid": spotify_id,
        "mp3": audio_path,
        "spectrogram": str(spectro_path)
    })

# build the DataFrame with proper columns
spectrograms = pd.DataFrame(rows, columns=["spotifyid", "mp3", "spectrogram"])
spectrograms.to_csv("DataSets/Spectrograms.csv", index=False, encoding="utf-8")