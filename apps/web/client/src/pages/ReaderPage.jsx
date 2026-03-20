import { useState, useEffect, useRef } from 'react';
import { useParams, useLocation } from 'react-router-dom';
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
        <h2 style={{ marginBottom: '0.5rem' }}>{location.state?.title || 'Your Text'}</h2>
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
            const bgColor = hexToRgba(color, 0.25);
            return (
              <span
                key={i}
                style={{
                  backgroundColor: bgColor,
                  padding: '0.1em 0.3em',
                  borderRadius: '4px',
                  cursor: 'default',
                }}
                title={`${seg.dominant} | ${seg.matchedSong?.song_name || ''}`}
                onClick={() => setActiveIndex(i)}
              >
                {seg.text}{' '}
              </span>
            );
          })}
        </div>

        <AudioPlayer song={null} />
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
