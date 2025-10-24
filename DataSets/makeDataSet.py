import pandas as pd
from pathlib import Path
import os
from SpotifyToSpectrogram.get_metadata import get_data_from_id
from SpotifyToSpectrogram.name_to_audio import download_mp3_from_spotify_id
from SpotifyToSpectrogram.mp3ToSpectrogram import plot_spectrogram_and_save


IMG_OUTPUT_PATH = Path("DataSets/img")

df = pd.read_csv('DataSets/SpotifyMetaToGems_Final.csv')
spectrograms = pd.DataFrame(columns=['spotifyid, mp3, spectrogram'])

head = df.head(1)
for id in head['spotifyid']:
    track = get_data_from_id(id)
    
    # spectrograms['spotifyid'] = id
    audio_path = download_mp3_from_spotify_id(track)

    audio_path = os.path.normpath(audio_path).replace("\\", "/")
    
    plot_spectrogram_and_save(audio_path, IMG_OUTPUT_PATH / f".png")