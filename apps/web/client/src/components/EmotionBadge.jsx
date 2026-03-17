import { getEmotionColor } from '../utils/emotions';

export default function EmotionBadge({ emotion, value }) {
  return (
    <span
      className="emotion-badge"
      style={{ backgroundColor: getEmotionColor(emotion) }}
    >
      {emotion}{value != null ? ` ${Math.round(value * 100)}%` : ''}
    </span>
  );
}
