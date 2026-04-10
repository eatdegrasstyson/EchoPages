import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useSpotify } from '../context/SpotifyContext';

export default function CallbackPage() {
  const navigate = useNavigate();
  const { storeTokens } = useSpotify();

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const accessToken = params.get('access_token');
    const refreshToken = params.get('refresh_token');
    const expiresIn = Number(params.get('expires_in')) || 3600;
    const error = params.get('error');

    if (error || !accessToken) {
      console.error('Spotify auth error:', error);
      navigate('/');
      return;
    }

    storeTokens(accessToken, refreshToken, expiresIn);
    navigate('/');
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return <div className="loading">Connecting to Spotify...</div>;
}
