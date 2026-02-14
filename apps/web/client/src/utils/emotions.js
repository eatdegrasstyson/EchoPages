// GEMs emotion categories with display colors
export const GEMS_COLORS = {
  Wonder:        '#f39c12',
  Transcendence: '#9b59b6',
  Tenderness:    '#e91e63',
  Nostalgia:     '#8d6e63',
  Peacefulness:  '#2ecc71',
  Power:         '#e74c3c',
  Joy:           '#f1c40f',
  Tension:       '#34495e',
  Sadness:       '#3498db',
};

export const GEMS = Object.keys(GEMS_COLORS);

/**
 * Return the dominant emotion from a GEMs vector object.
 */
export function getDominantEmotion(emotions) {
  if (!emotions) return null;
  const entries = emotions instanceof Map
    ? [...emotions.entries()]
    : Object.entries(emotions);
  if (entries.length === 0) return null;
  return entries.reduce((best, [key, val]) =>
    val > best[1] ? [key, val] : best
  )[0];
}

/**
 * Get the color for a given emotion name.
 */
export function getEmotionColor(emotion) {
  return GEMS_COLORS[emotion] || '#666';
}
