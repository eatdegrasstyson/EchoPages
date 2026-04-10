import { useState } from 'react';
import { useSpotify } from '../context/SpotifyContext';

export default function AudioPlayer() {
  const { isConnected, isReady, isPaused, currentTrack, togglePlay, setVolume } = useSpotify();
  const [volume, setVolumeState] = useState(0.7);

  if (!isConnected) return null;

  function handleVolumeChange(e) {
    const val = Number(e.target.value);
    setVolumeState(val);
    setVolume(val);
  }

  const artistNames = currentTrack?.artists?.map((a) => a.name).join(', ') ?? '';

  return (
    <div className="audio-player">
      <div className="audio-player-info">
        {currentTrack ? (
          <>
            <div className="audio-player-title">{currentTrack.name}</div>
            <div className="audio-player-artist">{artistNames}</div>
          </>
        ) : (
          <div className="audio-player-title" style={{ color: 'var(--text-secondary)' }}>
            {isReady ? 'No track playing' : 'Connecting to Spotify...'}
          </div>
        )}
      </div>

      <button
        className="btn btn-secondary"
        onClick={togglePlay}
        disabled={!isReady || !currentTrack}
        style={{ minWidth: '88px' }}
      >
        {isPaused ? '▶ Play' : '⏸ Pause'}
      </button>

      <div className="audio-volume">
        <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>🔊</span>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={volume}
          onChange={handleVolumeChange}
          style={{ width: '80px', accentColor: 'var(--accent)' }}
        />
      </div>
    </div>
  );
}
