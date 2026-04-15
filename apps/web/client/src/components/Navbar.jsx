import { Link } from 'react-router-dom';
import { useSpotify } from '../context/SpotifyContext';

export default function Navbar() {
  const { isConnected, isReady, login, logout } = useSpotify();

  return (
    <nav className="navbar">
      <Link to="/" className="navbar-brand">EchoPages</Link>
      <ul className="navbar-links">
        <li><Link to="/">Home</Link></li>
        <li><Link to="/upload">Upload</Link></li>
        <li>
          {isConnected ? (
            <button className="btn btn-secondary" onClick={logout} style={{ fontSize: '0.85rem', padding: '0.4rem 1rem' }}>
              {isReady ? '● ' : '○ '}Spotify Connected
            </button>
          ) : (
            <button className="btn btn-primary" onClick={login} style={{ fontSize: '0.85rem', padding: '0.4rem 1rem' }}>
              Connect Spotify
            </button>
          )}
        </li>
      </ul>
    </nav>
  );
}
