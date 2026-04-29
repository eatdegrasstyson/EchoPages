import { getDominantEmotion, getEmotionColor } from '../utils/emotions';

export default function EmotionTimeline({ segments, activeIndex, onSegmentClick }) {
  if (!segments || segments.length === 0) return null;

  return (
    <div style={{ marginBottom: '1.5rem' }}>
      <div className="emotion-timeline" title="Click a segment to jump to it">
        {segments.map((seg, i) => {
          const dominant = seg.dominant || getDominantEmotion(seg.emotions);
          const isActive = activeIndex === i;
          return (
            <div
              key={i}
              className="emotion-timeline-segment"
              style={{
                flex: 1,
                backgroundColor: getEmotionColor(dominant),
                opacity: isActive ? 1 : 0.55,
                transform: isActive ? 'scaleY(1.3)' : 'scaleY(1)',
                transition: 'opacity 0.15s, transform 0.15s',
              }}
              title={`#${i + 1}: ${dominant || 'unknown'}`}
              onClick={() => onSegmentClick(i)}
            />
          );
        })}
      </div>
      <div style={{ textAlign: 'right', fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
        {segments.length} sentence{segments.length !== 1 ? 's' : ''}
      </div>
    </div>
  );
}
