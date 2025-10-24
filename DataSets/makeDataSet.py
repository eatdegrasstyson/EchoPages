import pandas as pd
from SpotifyToSpectrogram.get_metadata import get_data_from_id
from SpotifyToSpectrogram.name_to_audio import download_mp3_from_spotify_id

df = pd.read_csv('DataSets/SpotifyMetaToGems_Final.csv')

spectrograms = pd.DataFrame(columns=['spotifyid, artist, name, mp3, spectrogram'])

head = df.head(1)
for id in head['spotifyid']:
    track = get_data_from_id(id)

    # artist = track[0]
    # song = track[1]
    
    # spectrograms['spotifyid'] = id
    # spectrograms['name'] = song

    download_mp3_from_spotify_id(track)