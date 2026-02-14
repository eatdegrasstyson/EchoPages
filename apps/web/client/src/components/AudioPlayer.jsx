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

  return (
    <div className="audio-player">
      <div className="audio-player-info">
        <div className="audio-player-title">{song.title}</div>
        <div className="audio-player-artist">{song.artist}</div>
      </div>
      <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
        Audio playback coming soon
      </div>
    </div>
  );
}
