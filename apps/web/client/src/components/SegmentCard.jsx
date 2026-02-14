import EmotionBadge from './EmotionBadge';
import { getDominantEmotion, getEmotionColor } from '../utils/emotions';

export default function SegmentCard({ segment, isActive, onClick }) {
  const dominant = getDominantEmotion(segment.emotions);
  const song = segment.matchedSong;

  return (
    <div
      className={`card segment-card${isActive ? ' active' : ''}`}
      style={{ borderLeftColor: getEmotionColor(dominant) }}
      onClick={onClick}
    >
      <div className="segment-card-header">
        <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
          #{segment.index + 1}
        </span>
        {dominant && (
          <EmotionBadge
            emotion={dominant}
            value={segment.emotions[dominant]}
          />
        )}
      </div>
      <p>{segment.content}</p>
      {song && (
        <div className="segment-card-song">
          {song.title} &mdash; {song.artist}
        </div>
      )}
    </div>
  );
}
