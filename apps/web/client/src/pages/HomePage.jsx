import { Link } from 'react-router-dom';

const steps = [
  { title: 'Paste your text', desc: 'Upload a book chapter, story, poem, or article.' },
  { title: 'Emotion analysis', desc: 'Each passage is analyzed for its emotional tone.' },
  { title: 'Music matches', desc: 'A soundtrack is matched to every segment in real time.' },
];

export default function HomePage() {
  return (
    <div>
      <div className="home-hero">
        <h1>EchoPages</h1>
        <p>
          An immersive reading experience where music dynamically matches
          the emotional tone of what you read.
        </p>
        <Link to="/upload" className="btn btn-primary">
          Get Started
        </Link>
      </div>

      <div className="home-steps">
        {steps.map((step, i) => (
          <div key={i} className="card home-step">
            <div className="home-step-number">{i + 1}</div>
            <h3>{step.title}</h3>
            <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
              {step.desc}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
