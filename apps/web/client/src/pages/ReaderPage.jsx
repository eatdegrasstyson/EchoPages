import { useState, useEffect, useRef } from 'react';
import { useParams } from 'react-router-dom';
import { mockTexts } from '../data/mockData';
import EmotionTimeline from '../components/EmotionTimeline';
import SegmentCard from '../components/SegmentCard';
import AudioPlayer from '../components/AudioPlayer';

export default function ReaderPage() {
  const { id } = useParams();
  const [text, setText] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeIndex, setActiveIndex] = useState(0);
  const segmentRefs = useRef([]);

  useEffect(() => {
    // Simulate loading from mock data
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
