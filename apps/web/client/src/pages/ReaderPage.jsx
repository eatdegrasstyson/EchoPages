import { useState, useEffect, useRef } from 'react';
import { useParams, useLocation, useNavigate } from 'react-router-dom';
import { mockTexts } from '../data/mockData';
import EmotionTimeline from '../components/EmotionTimeline';
import SegmentCard from '../components/SegmentCard';
import AudioPlayer from '../components/AudioPlayer';
import { GEMS_COLORS } from '../utils/emotions';

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export default function ReaderPage() {
  const { id } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const [text, setText] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeIndex, setActiveIndex] = useState(0);
  const segmentRefs = useRef([]);

  const apiSegments = location.state?.segments || null;

  useEffect(() => {
    if (apiSegments) {
      setLoading(false);
      return;
    }
    setTimeout(() => {
      setText(mockTexts[id] || mockTexts['1']);
      setLoading(false);
    }, 300);
  }, [id]);

  function scrollToSegment(index) {
    setActiveIndex(index);
    segmentRefs.current[index]?.scrollIntoView({
      behavior: 'smooth',
      block: 'center',
    });
  }

  if (loading) return <div className="loading">Loading...</div>;

  // API flow: render prose block with inline colored spans
  if (apiSegments) {
    return (
      <div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
          <h2>{location.state?.title || 'Your Text'}</h2>
          <button className="btn btn-secondary" style={{ fontSize: '0.85rem' }} onClick={() => navigate('/upload')}>
            Analyze new text
          </button>
        </div>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
          {apiSegments.length} sentences
        </p>

        <EmotionTimeline
          segments={apiSegments}
          activeIndex={activeIndex}
          onSegmentClick={setActiveIndex}
        />

        <div className="card" style={{ lineHeight: '2', fontSize: '1.05rem', marginTop: '1.5rem' }}>
          {apiSegments.map((seg, i) => {
            const color = GEMS_COLORS[seg.dominant] || '#666666';
            const bgColor = hexToRgba(color, activeIndex === i ? 0.45 : 0.2);
            const outline = activeIndex === i ? `2px solid ${color}` : 'none';
            return (
              <span
                key={i}
                style={{
                  backgroundColor: bgColor,
                  padding: '0.15em 0.35em',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  outline,
                  outlineOffset: '1px',
                  transition: 'background-color 0.2s, outline 0.2s',
                }}
                title={`${seg.dominant} | ${seg.matchedSong?.song_name || ''}`}
                onClick={() => setActiveIndex(i)}
              >
                {seg.text}{' '}
              </span>
            );
          })}
        </div>

        {apiSegments[activeIndex] && (
          <div className="emotion-panel">
            <div className="emotion-panel-title">
              <span className="emotion-badge" style={{ backgroundColor: GEMS_COLORS[apiSegments[activeIndex].dominant] || '#666' }}>
                {apiSegments[activeIndex].dominant}
              </span>
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginLeft: '0.5rem' }}>
                {apiSegments[activeIndex].matchedSong?.song_name || ''}
              </span>
            </div>
            <div className="emotion-bars">
              {Object.entries(apiSegments[activeIndex].emotions)
                .sort(([, a], [, b]) => b - a)
                .map(([emotion, score]) => (
                  <div key={emotion} className="emotion-bar-row">
                    <span className="emotion-bar-label">{emotion}</span>
                    <div className="emotion-bar-track">
                      <div
                        className="emotion-bar-fill"
                        style={{ width: `${Math.round(score * 100)}%`, backgroundColor: GEMS_COLORS[emotion] || '#666' }}
                      />
                    </div>
                    <span className="emotion-bar-pct">{Math.round(score * 100)}%</span>
                  </div>
                ))}
            </div>
          </div>
        )}

        <AudioPlayer song={apiSegments[activeIndex]?.matchedSong || null} />
      </div>
    );
  }

  // Mock data flow: existing behavior
  if (!text) return <div className="card">Text not found.</div>;

  const activeSong = text.segments[activeIndex]?.matchedSong || null;

  return (
    <div>
      <h2 style={{ marginBottom: '0.5rem' }}>{text.title}</h2>
      <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
        {text.segments.length} segments
      </p>

      <EmotionTimeline
        segments={text.segments}
        activeIndex={activeIndex}
        onSegmentClick={scrollToSegment}
      />

      <div>
        {text.segments.map((seg, i) => (
          <div key={i} ref={(el) => (segmentRefs.current[i] = el)}>
            <SegmentCard
              segment={seg}
              isActive={activeIndex === i}
              onClick={() => setActiveIndex(i)}
            />
          </div>
        ))}
      </div>

      <AudioPlayer song={activeSong} />
    </div>
  );
}
