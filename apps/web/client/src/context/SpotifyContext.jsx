import { createContext, useContext, useState, useEffect, useRef } from 'react';

const SpotifyContext = createContext(null);

export function useSpotify() {
  return useContext(SpotifyContext);
}

export function SpotifyProvider({ children }) {
  const [accessToken, setAccessToken] = useState(() => localStorage.getItem('spotify_access_token'));
  const [refreshToken, setRefreshToken] = useState(() => localStorage.getItem('spotify_refresh_token'));
  const [expiresAt, setExpiresAt] = useState(() => Number(localStorage.getItem('spotify_expires_at')) || 0);
  const [deviceId, setDeviceId] = useState(null);
  const [isReady, setIsReady] = useState(false);
  const [isPaused, setIsPaused] = useState(true);
  const [currentTrack, setCurrentTrack] = useState(null);

  // Keep refs in sync so async callbacks always see the latest values
  const accessTokenRef = useRef(accessToken);
  const refreshTokenRef = useRef(refreshToken);
  const expiresAtRef = useRef(expiresAt);
  const playerRef = useRef(null);

  useEffect(() => { accessTokenRef.current = accessToken; }, [accessToken]);
  useEffect(() => { refreshTokenRef.current = refreshToken; }, [refreshToken]);
  useEffect(() => { expiresAtRef.current = expiresAt; }, [expiresAt]);

  function storeTokens(access, refresh, expiresIn) {
    const exp = Date.now() + Number(expiresIn) * 1000;
    setAccessToken(access);
    setExpiresAt(exp);
    localStorage.setItem('spotify_access_token', access);
    localStorage.setItem('spotify_expires_at', exp);
    if (refresh) {
      setRefreshToken(refresh);
      localStorage.setItem('spotify_refresh_token', refresh);
    }
  }

  async function getValidToken() {
    // If token is still good (with 60s buffer), return it
    if (accessTokenRef.current && Date.now() < expiresAtRef.current - 60000) {
      return accessTokenRef.current;
    }
    // Otherwise refresh
    if (!refreshTokenRef.current) return null;
    try {
      const res = await fetch('/api/auth/refresh', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: refreshTokenRef.current }),
      });
      const data = await res.json();
      if (!data.access_token) return null;
      storeTokens(data.access_token, null, data.expires_in);
      return data.access_token;
    } catch {
      return null;
    }
  }

  // Initialize the Web Playback SDK whenever we have a token
  useEffect(() => {
    if (!accessToken) return;

    function initPlayer() {
      if (playerRef.current) return; // already initialized

      const player = new window.Spotify.Player({
        name: 'EchoPages',
        getOAuthToken: (cb) => {
          getValidToken().then(cb);
        },
        volume: 0.7,
      });

      player.addListener('ready', ({ device_id }) => {
        setDeviceId(device_id);
        setIsReady(true);
      });

      player.addListener('not_ready', () => {
        setIsReady(false);
      });

      player.addListener('player_state_changed', (state) => {
        if (!state) return;
        setIsPaused(state.paused);
        setCurrentTrack(state.track_window?.current_track ?? null);
      });

      player.connect();
      playerRef.current = player;
    }

    if (window.Spotify) {
      initPlayer();
    } else {
      window.onSpotifyWebPlaybackSDKReady = initPlayer;
    }

    return () => {
      if (playerRef.current) {
        playerRef.current.disconnect();
        playerRef.current = null;
      }
    };
  }, [!!accessToken]); // eslint-disable-line react-hooks/exhaustive-deps

  function login() {
    // Full page navigation — goes to Flask which redirects to Spotify
    window.location.href = 'http://localhost:5000/api/auth/login';
  }

  function logout() {
    if (playerRef.current) {
      playerRef.current.disconnect();
      playerRef.current = null;
    }
    setAccessToken(null);
    setRefreshToken(null);
    setExpiresAt(0);
    setDeviceId(null);
    setIsReady(false);
    setIsPaused(true);
    setCurrentTrack(null);
    localStorage.removeItem('spotify_access_token');
    localStorage.removeItem('spotify_refresh_token');
    localStorage.removeItem('spotify_expires_at');
  }

  // Play a track (by Spotify track ID) starting at a given second offset
  async function playTrack(spotifyId, startSeconds = 0) {
    const token = await getValidToken();
    if (!token || !deviceId) return;

    await fetch(`https://api.spotify.com/v1/me/player/play?device_id=${deviceId}`, {
      method: 'PUT',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        uris: [`spotify:track:${spotifyId}`],
        position_ms: Math.round(startSeconds * 1000),
      }),
    });
  }

  async function togglePlay() {
    if (!playerRef.current) return;
    await playerRef.current.togglePlay();
  }

  async function setVolume(fraction) {
    if (!playerRef.current) return;
    await playerRef.current.setVolume(fraction);
  }

  return (
    <SpotifyContext.Provider value={{
      isConnected: !!accessToken,
      isReady,
      isPaused,
      currentTrack,
      deviceId,
      storeTokens,
      login,
      logout,
      playTrack,
      togglePlay,
      setVolume,
    }}>
      {children}
    </SpotifyContext.Provider>
  );
}
