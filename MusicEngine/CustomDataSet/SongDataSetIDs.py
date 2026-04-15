import os
import sys
import json
import csv
import time
import base64
import requests
from SpotifyToSpectrogram import SpotifyCreds
# ── Spotify Auth ──────────────────────────────────────────────────────────────

def get_spotify_token(client_id: str, client_secret: str) -> str:
    """Get a Spotify access token using Client Credentials flow."""
    auth_str = f"{client_id}:{client_secret}"
    b64_auth = base64.b64encode(auth_str.encode()).decode()
    resp = requests.post(
        "https://accounts.spotify.com/api/token",
        headers={"Authorization": f"Basic {b64_auth}"},
        data={"grant_type": "client_credentials"},
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


# ── Search Helpers ────────────────────────────────────────────────────────────

def search_tracks(query: str, token: str, limit: int = 5) -> list[dict]:
    """Search Spotify for tracks matching a query."""
    resp = requests.get(
        "https://api.spotify.com/v1/search",
        headers={"Authorization": f"Bearer {token}"},
        params={"q": query, "type": "track", "limit": limit},
    )
    resp.raise_for_status()
    items = resp.json().get("tracks", {}).get("items", [])
    results = []
    for t in items:
        results.append({
            "song_name": t["name"],
            "artist": ", ".join(a["name"] for a in t["artists"]),
            "spotify_id": t["id"],
            "spotify_uri": t["uri"],
        })
    return results


# ── GEMS-9 Search Queries ────────────────────────────────────────────────────
# Each emotion has curated search queries designed to surface representative songs.
# Multiple queries per emotion help diversify results beyond a single keyword.

GEMS9_QUERIES: dict[str, list[str]] = {
    "Wonder": [
        "epic orchestral wonder",
        "awe inspiring cinematic",
        "Sigur Ros",
        "Explosions in the Sky",
        "Hans Zimmer Interstellar",
        "Pink Floyd Comfortably Numb",
        "Radiohead OK Computer",
        "Godspeed You Black Emperor",
    ],
    "Transcendence": [
        "spiritual transcendence music",
        "sacred choral music",
        "Arvo Part Spiegel",
        "Nusrat Fateh Ali Khan",
        "Gregorian chant",
        "Brian Eno ambient",
        "Sufi devotional music",
        "Lisa Gerrard Dead Can Dance",
    ],
    "Tenderness": [
        "tender love ballad",
        "gentle acoustic love",
        "Jeff Buckley Hallelujah",
        "Bon Iver Skinny Love",
        "Norah Jones Come Away With Me",
        "Adele Make You Feel My Love",
        "Iron and Wine Flightless Bird",
        "Nick Drake Pink Moon",
    ],
    "Nostalgia": [
        "nostalgic bittersweet song",
        "memories childhood song",
        "Simon and Garfunkel Sound of Silence",
        "Fleetwood Mac Landslide",
        "Cat Stevens Father and Son",
        "Eric Clapton Tears in Heaven",
        "Green Day Time of Your Life",
        "The Cranberries Dreams",
    ],
    "Peacefulness": [
        "peaceful calm instrumental",
        "serene nature ambient",
        "Debussy Clair de Lune",
        "Marconi Union Weightless",
        "Enya Only Time",
        "Jack Johnson Banana Pancakes",
        "Nils Frahm Says",
        "Ludovico Einaudi Nuvole Bianche",
    ],
    "Power": [
        "powerful epic anthem",
        "energetic metal power",
        "Queen We Will Rock You",
        "Rage Against the Machine Killing",
        "AC DC Thunderstruck",
        "Muse Knights of Cydonia",
        "Two Steps From Hell Victory",
        "Led Zeppelin Immigrant Song",
    ],
    "Joyful Activation": [
        "happy upbeat dance pop",
        "feel good summer song",
        "Pharrell Happy",
        "Earth Wind Fire September",
        "Mark Ronson Uptown Funk",
        "Katrina and the Waves Walking on Sunshine",
        "Lizzo Good as Hell",
        "Daft Punk Get Lucky",
    ],
    "Tension": [
        "tense suspenseful music",
        "dark intense electronic",
        "Nine Inch Nails Closer",
        "Massive Attack Angel",
        "Radiohead Climbing Up the Walls",
        "Tool Lateralus",
        "Portishead Sour Times",
        "Hans Zimmer Dark Knight",
    ],
    "Sadness": [
        "sad melancholic piano",
        "heartbreak grief song",
        "Radiohead Exit Music",
        "Johnny Cash Hurt",
        "Samuel Barber Adagio for Strings",
        "Billie Eilish When the Party Over",
        "Elliott Smith Between the Bars",
        "Beethoven Moonlight Sonata",
    ],
}

SONGS_PER_EMOTION = 12  # ceil(100/9) ≈ 12, we'll trim to 100 at the end


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    creds = SpotifyCreds.SpotifyCreds()
    client_id = creds.CLIENT
    client_secret = creds.CLIENT_SECRET

    if not client_id or not client_secret:
        print("ERROR: Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET env vars.")
        print("  export SPOTIFY_CLIENT_ID='your_id'")
        print("  export SPOTIFY_CLIENT_SECRET='your_secret'")
        sys.exit(1)

    print("Authenticating with Spotify...")
    token = get_spotify_token(client_id, client_secret)
    print("Authenticated!\n")

    all_songs = []
    seen_ids = set()

    for emotion, queries in GEMS9_QUERIES.items():
        print(f"── {emotion} ──")
        emotion_songs = []

        for query in queries:
            if len(emotion_songs) >= SONGS_PER_EMOTION:
                break

            results = search_tracks(query, token, limit=3)
            for track in results:
                if track["spotify_id"] not in seen_ids and len(emotion_songs) < SONGS_PER_EMOTION:
                    track["gems9_emotion"] = emotion
                    emotion_songs.append(track)
                    seen_ids.add(track["spotify_id"])
                    print(f"  + {track['song_name']} — {track['artist']}")

            time.sleep(0.1)  # gentle rate limiting

        all_songs.extend(emotion_songs)
        print(f"  ({len(emotion_songs)} songs)\n")

    # ── Save CSV ──
    csv_path = "gems9_songs.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "gems9_emotion", "song_name", "artist", "spotify_id", "spotify_uri"
        ])
        writer.writeheader()
        writer.writerows(all_songs)
    print(f"Saved: {csv_path}")

    # ── Save JSON ──
    json_path = "gems9_songs.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_songs, f, indent=2, ensure_ascii=False)
    print(f"Saved: {json_path}")

    # ── Print summary table ──
    print("\n── Summary ──")
    from collections import Counter
    counts = Counter(s["gems9_emotion"] for s in all_songs)
    for emotion in GEMS9_QUERIES:
        print(f"  {emotion:>20s}: {counts.get(emotion, 0)} songs")
    print(f"  {'TOTAL':>20s}: {len(all_songs)} songs")


if __name__ == "__main__":
    main()