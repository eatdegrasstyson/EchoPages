import spotipy
from spotipy.oauth2 import SpotifyOAuth
from SpotifyToSpectrogram import SpotifyCreds
from spotipy.oauth2 import SpotifyClientCredentials


scope = "user-library-read"
creds = SpotifyCreds.SpotifyCreds()
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=creds.CLIENT,
                                               client_secret=creds.CLIENT_SECRET,
                                               redirect_uri="http://127.0.0.1:3000",
                                               scope="user-library-read"))

def get_data_from_id(spotify_id):
    """
    Takes a Spotify ID (track, album, or artist) and returns the meta_data.
    """
    try:
        # Try fetching as a track
        track = sp.track(spotify_id)
        return [track['artists'][0]['name'], track['name'], track['duration_ms']]

    except spotipy.SpotifyException:
        return "Invalid ID or unsupported type"
    
def get_audio_features(spotify_id):
    try:
        features = sp.audio_features([spotify_id])[0]
        if features is None:
            return "Invalid track ID or features unavailable"
        
        #Extract commonly used audio features
        return {
            "acousticness": features["acousticness"],
            "danceability": features["danceability"],
            "energy": features["energy"],
            "instrumentalness": features["instrumentalness"],
            "liveness": features["liveness"],
            "loudness": features["loudness"],
            "speechiness": features["speechiness"],
            "tempo": features["tempo"],
            "valence": features["valence"],
            "key": features["key"],
            "mode": features["mode"],
            "time_signature": features["time_signature"],
            "duration_ms": features["duration_ms"]
        }
    
    except spotipy.SpotifyException as e:
        return f"Spotify API error: {str(e)}"

# Example usage:
if __name__ == "__main__":
    test_id = "3n3Ppam7vgaVa1iaRUc9Lp"  # Example: "Mr. Brightside" by The Killers
    print(get_data_from_id(test_id))