// Mock data for the reader
export const mockTexts = {
  '1': {
    _id: '1',
    title: 'Sample Story',
    segments: [
      {
        index: 0,
        content: 'Once upon a time, in a land far away, there lived a young adventurer who dreamed of exploring the world.',
        emotions: {
          Wonder: 0.8,
          Joy: 0.6,
          Peacefulness: 0.3,
          Transcendence: 0.2,
          Tenderness: 0.1,
          Nostalgia: 0.1,
          Power: 0.0,
          Tension: 0.0,
          Sadness: 0.0,
        },
        matchedSong: {
          title: 'Adventure Awaits',
          artist: 'Sample Artist',
          previewUrl: null,
        },
      },
      {
        index: 1,
        content: 'The journey was long and filled with challenges, but the adventurer pressed on with determination.',
        emotions: {
          Power: 0.7,
          Tension: 0.5,
          Joy: 0.3,
          Wonder: 0.2,
          Transcendence: 0.1,
          Tenderness: 0.0,
          Nostalgia: 0.0,
          Peacefulness: 0.0,
          Sadness: 0.0,
        },
        matchedSong: {
          title: 'Brave Hearts',
          artist: 'Sample Artist 2',
          previewUrl: null,
        },
      },
      {
        index: 2,
        content: 'At the end of the journey, the adventurer found something unexpected: a sense of peace and belonging.',
        emotions: {
          Peacefulness: 0.8,
          Tenderness: 0.7,
          Nostalgia: 0.4,
          Joy: 0.3,
          Wonder: 0.2,
          Transcendence: 0.1,
          Power: 0.0,
          Tension: 0.0,
          Sadness: 0.0,
        },
        matchedSong: {
          title: 'Coming Home',
          artist: 'Sample Artist 3',
          previewUrl: null,
        },
      },
    ],
  },
};
