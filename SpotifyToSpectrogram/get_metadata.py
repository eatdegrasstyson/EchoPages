import spotipy
from spotipy.oauth2 import SpotifyOAuth
from SpotifyToSpectrogram import SpotifyCreds

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

# Example usage:
if __name__ == "__main__":
    test_id = "3n3Ppam7vgaVa1iaRUc9Lp"  # Example: "Mr. Brightside" by The Killers
    print(get_data_from_id(test_id))