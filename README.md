# EchoPages

EchoPages is a multi-modal analysis platform that bridges music and text through emotional intelligence. It analyzes audio features and text sentiment to match written content with emotionally resonant music in real time.

## Project Structure

```
echopages/
├── MusicEngine/              # Audio processing and ML model
│   ├── SpotifyToMeta/        # Scripts to aggregate music metadata
│   ├── SpotifyToSpectrogram/ # Audio-to-spectrogram conversion
│   ├── Model/                # CRNN training and emotion prediction
│   └── DataSets/             # Raw and processed datasets
├── TextEngine/               # Text emotion analysis
│   ├── GoEmotions/           # GoEmotions-based classifier
│   ├── GoEmotionsRoBERTa/    # RoBERTa fine-tuned classifier
│   ├── NRCLexTesting/        # Lexicon-based emotion scoring
│   └── TransformersTesting/  # Transformer model experiments
└── apps/web/                 # Full-stack web application
    ├── client/               # React + Vite frontend
    └── server/               # Flask backend API
```

## Getting Started

### ML Pipeline (Python)

See [SETUP.md](SETUP.md) for full instructions to set up the Python environment, configure Spotify credentials, and run emotion prediction on songs.

### Web App

**Backend (Flask):**
```bash
source venv/bin/activate
python apps/web/server/app.py
```

**Frontend (React + Vite):**
```bash
cd apps/web/client
npm install
npm run dev
```

The frontend runs at `http://localhost:5173` and the backend at `http://localhost:5000`.

## Prerequisites

- Python 3.8+
- Node.js 18+
- FFmpeg
- Spotify Developer Account (for ML pipeline)
