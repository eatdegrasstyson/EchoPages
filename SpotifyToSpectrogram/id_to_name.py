import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials


scope = "user-library-read"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="CLIENT_ID",
                                               client_secret="CLIENT_SECRET",
                                               redirect_uri="http://127.0.0.1:3000",
                                               scope="user-library-read"))

def get_name_from_id(spotify_id: str) -> str:
    """
    Takes a Spotify ID (track, album, or artist) and returns the name.
    """
    try:
        # Try fetching as a track
        track = sp.track(spotify_id)
        return track['name']
    except spotipy.SpotifyException:
        try:
            # Try fetching as an album
            album = sp.album(spotify_id)
            return album['name']
        except spotipy.SpotifyException:
            try:
                # Try fetching as an artist
                artist = sp.artist(spotify_id)
                return artist['name']
            except spotipy.SpotifyException:
                return "Invalid ID or unsupported type"

# Example usage:
if __name__ == "__main__":
    test_id = "3n3Ppam7vgaVa1iaRUc9Lp"  # Example: "Mr. Brightside" by The Killers
    print(get_name_from_id(test_id))
