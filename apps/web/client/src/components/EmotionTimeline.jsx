import { getDominantEmotion, getEmotionColor } from '../utils/emotions';

export default function EmotionTimeline({ segments, activeIndex, onSegmentClick }) {
  if (!segments || segments.length === 0) return null;

  return (
    <div className="emotion-timeline" title="Emotion arc">
      {segments.map((seg, i) => {
        const dominant = getDominantEmotion(seg.emotions);
        return (
          <div
            key={i}
            className="emotion-timeline-segment"
            style={{
              flex: 1,
              backgroundColor: getEmotionColor(dominant),
              opacity: activeIndex === i ? 1 : 0.6,
            }}
            title={`#${i + 1}: ${dominant || 'unknown'}`}
            onClick={() => onSegmentClick(i)}
          />
        );
      })}
    </div>
  );
}
