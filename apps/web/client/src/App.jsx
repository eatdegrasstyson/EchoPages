import { Routes, Route } from 'react-router-dom';
import { SpotifyProvider } from './context/SpotifyContext';
import Navbar from './components/Navbar';
import HomePage from './pages/HomePage';
import UploadPage from './pages/UploadPage';
import ReaderPage from './pages/ReaderPage';
import CallbackPage from './pages/CallbackPage';

export default function App() {
  return (
    <SpotifyProvider>
      <div className="app">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/read/:id" element={<ReaderPage />} />
            <Route path="/callback" element={<CallbackPage />} />
          </Routes>
        </main>
      </div>
    </SpotifyProvider>
  );
}
