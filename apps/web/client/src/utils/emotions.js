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


export function getTop3Emotions(emotions) {
  return Object.entries(emotions)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3);
}

// Blend the top 3 colors weighted by their emotion values
export function getTop3Color(emotions) {
  const top3 = getTop3Emotions(emotions);
  let r = 0, g = 0, b = 0, total = 0;

  top3.forEach(([key, value]) => {
    const hex = GEMS_COLORS[key] || '#666666';
    const weight = value;
    total += weight;
    r += parseInt(hex.slice(1, 3), 16) * weight;
    g += parseInt(hex.slice(3, 5), 16) * weight;
    b += parseInt(hex.slice(5, 7), 16) * weight;
  });

  r = Math.round(r / total);
  g = Math.round(g / total);
  b = Math.round(b / total);

  return `rgb(${r}, ${g}, ${b})`;
}

// Format top 3 for tooltip
export function formatTop3Tooltip(emotions) {
  return getTop3Emotions(emotions)
    .map(([key, val]) => `${key}: ${val.toFixed(2)}`)
    .join('\n');
}