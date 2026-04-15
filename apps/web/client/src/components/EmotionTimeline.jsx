import { getTop3Color, formatTop3Tooltip } from '../utils/emotions';

export default function EmotionTimeline({ segments, activeIndex, onSegmentClick }) {
  if (!segments || segments.length === 0) return null;

  return (
    <div className="emotion-timeline" title="Emotion arc">
      {segments.map((seg, i) => {
        const color = getTop3Color(seg.emotions);
        const tooltip = `#${i + 1}\n${formatTop3Tooltip(seg.emotions)}`;

        return (
          <div
            key={i}
            className="emotion-timeline-segment"
            style={{
              flex: 1,
              backgroundColor: color,
              opacity: activeIndex === i ? 1 : 0.6,
              cursor: 'pointer'
            }}
            title={tooltip}
            onClick={() => onSegmentClick(i)}
          />
        );
      })}
    </div>
  );
}