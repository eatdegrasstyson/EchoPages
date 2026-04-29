export default function AudioPlayer({ song }) {
  if (!song) {
    return (
      <div className="audio-player">
        <div className="audio-player-info">
          <div className="audio-player-title" style={{ color: 'var(--text-secondary)' }}>
            No track playing
          </div>
        </div>
      </div>
    );
  }

  const title = song.song_name || song.title || 'Unknown Track';
  const artist = song.artist || null;
  const spotifyID = song.spotifyID || null;

  return (
    <div className="audio-player">
      <div className="audio-player-info">
        <div className="audio-player-title">{title}</div>
        {artist && <div className="audio-player-artist">{artist}</div>}
      </div>
      {spotifyID ? (
        <iframe
          className="spotify-embed"
          src={`https://open.spotify.com/embed/track/${spotifyID}?utm_source=generator&theme=0`}
          allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
          loading="lazy"
          title={title}
        />
      ) : (
        <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
          Audio playback coming soon
        </div>
      )}
    </div>
  );
}
