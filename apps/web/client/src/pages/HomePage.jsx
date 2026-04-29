import { Link } from 'react-router-dom';
import { GEMS_COLORS } from '../utils/emotions';

const steps = [
  { icon: '📖', title: 'Paste your text', desc: 'Upload a book chapter, story, poem, or article.' },
  { icon: '🎭', title: 'Emotion analysis', desc: 'Each sentence is mapped to one of 9 musical emotion categories.' },
  { icon: '🎵', title: 'Live soundtrack', desc: 'A Spotify track is matched to every sentence in real time.' },
];

const GEMS_PREVIEW = Object.entries(GEMS_COLORS);

export default function HomePage() {
  return (
    <div>
      <div className="home-hero">
        <h1>EchoPages</h1>
        <p>
          An immersive reading experience where music dynamically adapts
          to the emotional tone of every sentence you read.
        </p>
        <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}>
          <Link to="/upload" className="btn btn-primary">
            Try it now
          </Link>
          <Link to="/read/1" className="btn btn-secondary">
            See a demo
          </Link>
        </div>
      </div>

      <div className="home-steps">
        {steps.map((step, i) => (
          <div key={i} className="card home-step">
            <div style={{ fontSize: '1.8rem', marginBottom: '0.5rem' }}>{step.icon}</div>
            <div className="home-step-number">{i + 1}</div>
            <h3>{step.title}</h3>
            <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
              {step.desc}
            </p>
          </div>
        ))}
      </div>

      <div className="card" style={{ marginTop: '2rem' }}>
        <h3 style={{ marginBottom: '0.75rem' }}>9 musical emotions</h3>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
          {GEMS_PREVIEW.map(([emotion, color]) => (
            <span key={emotion} className="emotion-badge" style={{ backgroundColor: color }}>
              {emotion}
            </span>
          ))}
        </div>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '0.75rem' }}>
          Based on the GEMS (Geneva Emotional Music Scale) model of music-evoked emotion.
        </p>
      </div>
    </div>
  );
}
