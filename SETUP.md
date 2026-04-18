# EchoPages Setup Guide

## Prerequisites
- Python 3.8 or higher
- FFmpeg (for audio processing)
- Spotify Developer Account

## Installation Steps

### 1. Install System Dependencies

**Install Python venv and FFmpeg:**
```bash
sudo apt update
sudo apt install python3-venv python3-full ffmpeg
```

**macOS:**
```bash
brew install python3 ffmpeg
```

### 2. Create and Activate Virtual Environment

**Create virtual environment:**
```bash
python3 -m venv venv
```

**Activate virtual environment:**
```bash
# On Linux/macOS/WSL
source venv/bin/activate

# On Windows Command Prompt
venv\Scripts\activate.bat

# On Windows PowerShell
venv\Scripts\Activate.ps1
```

You should see `(venv)` at the beginning of your command prompt.

### 3. Install Python Dependencies

**With virtual environment activated:**
```bash
pip install -r requirements.txt
```

This will install:
- TensorFlow (deep learning)
- NumPy, Pandas (data processing)
- Librosa (audio processing)
- Spotipy (Spotify API)
- yt-dlp (YouTube downloads)
- And other dependencies

### 4. Configure Spotify API Credentials

1. Visit https://developer.spotify.com/dashboard
2. Log in with your Spotify account
3. Click "Create an App"
4. Fill in:
   - App Name: "EchoPages" (or any name)
   - App Description: "Music emotion detection"
5. After creating, note your **Client ID** and **Client Secret**
6. Click "Edit Settings" and add this Redirect URI:
   ```
   http://127.0.0.1:3000
   ```
7. Save the settings
8. Edit `SpotifyToSpectrogram/SpotifyCreds.py` and replace:
   - `your_spotify_client_id_here` with your Client ID
   - `your_spotify_client_secret_here` with your Client Secret

### 5. Verify Installation

**With virtual environment activated:**
```bash
python -c "import tensorflow, librosa, spotipy, yt_dlp; print('All packages installed successfully!')"
```

## Usage

**IMPORTANT:** Always activate the virtual environment before running any Python scripts:
```bash
source venv/bin/activate  # Linux/macOS/WSL
```

### Training the Model

If you have the dataset files ready:
```bash
python Model/train_CRNN.py
```

Required files:
- `DataSets/SpotifyIDMappings.csv`
- `DataSets/SpotifyMetaToGems_Final.csv`
- Spectrogram files (.npy) in the paths specified in the CSV

### Predicting Emotions

Predict emotions from any Spotify track:
```bash
python Model/PredictEmotionFromSong.py <spotify_id>
```

Example (Mr. Brightside by The Killers):
```bash
python Model/PredictEmotionFromSong.py 3n3Ppam7vgaVa1iaRUc9Lp
```

The script will:
1. Fetch track metadata from Spotify
2. Download the audio from YouTube
3. Convert to mel-spectrogram
4. Predict emotion probabilities for all 9 emotions:
   - Wonder, Transcendence, Tenderness, Nostalgia, Peacefulness
   - Power, Joy, Tension, Sadness
5. Display results

### Finding Spotify Track IDs

To get a Spotify track ID:
1. Open Spotify (desktop or web)
2. Right-click on a song
3. Select "Share" → "Copy Song Link"
4. The ID is the part after `track/` in the URL

Example: `https://open.spotify.com/track/3n3Ppam7vgaVa1iaRUc9Lp`
Track ID: `3n3Ppam7vgaVa1iaRUc9Lp`

## Running the Web App

With the virtual environment activated and dependencies installed, you can run the full web application:

**Start the Flask backend:**
```bash
source venv/bin/activate
python apps/web/server/app.py
```

The backend runs at `http://localhost:5000` and exposes the song-matching API.

**Start the React frontend (in a separate terminal):**
```bash
cd apps/web/client
npm install
npm run dev
```

The frontend runs at `http://localhost:5173`. Both services must be running for the full experience.

## Deactivating Virtual Environment

When you're done working:
```bash
deactivate
```

## Troubleshooting

### "externally-managed-environment" error
You need to use a virtual environment. See steps 1-3 above.

### "No module named 'tensorflow'"
Make sure your virtual environment is activated: `source venv/bin/activate`
Then run: `pip install -r requirements.txt`

### "No module named 'SpotifyToSpectrogram.SpotifyCreds'"
Make sure `SpotifyToSpectrogram/SpotifyCreds.py` exists and has valid credentials

### "ffmpeg not found"
Install FFmpeg: `sudo apt install ffmpeg`

### Spotify authentication errors
1. Double-check your credentials in `SpotifyCreds.py`
2. Verify the redirect URI is set to `http://127.0.0.1:3000`
3. Make sure your app is not in Development Mode restrictions
4. The first time you run a script, a browser will open for authentication

### Model not found
The prediction script expects a trained model at `Model/epoch_09.h5`. You need to either:
- Train the model using `train_CRNN.py`
- Or obtain a pre-trained model file

### Permission errors on WSL
If you get permission errors, you may need to:
```bash
chmod +x venv/bin/activate
```

## Quick Start Summary

```bash
# One-time setup
sudo apt update
sudo apt install python3-venv python3-full ffmpeg
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Edit SpotifyToSpectrogram/SpotifyCreds.py with your Spotify credentials

# Every time you work on the project
source venv/bin/activate

# Run predictions
python Model/PredictEmotionFromSong.py <spotify_id>

# When done
deactivate
```
