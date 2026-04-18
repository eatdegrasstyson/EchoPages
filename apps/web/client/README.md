# EchoPages Frontend

React + Vite frontend for the EchoPages immersive reading platform.

## Features

- **PDF Upload**: Upload PDF files for reading
- **Text Input**: Paste text directly into the reader
- **E-Reader Interface**: Clean reading experience with emotion-matched music
- **Live Song Matching**: Connects to the Flask backend to match text emotions to songs in real time

## Running the App

Start the Flask backend first (see [SETUP.md](../../../SETUP.md)), then:

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

The app runs at `http://localhost:5173`. The backend must be running at `http://localhost:5000` for song matching to work.

## Project Structure

- `src/pages/` - Main pages (Home, Upload, Reader)
- `src/components/` - Reusable UI components
- `src/data/` - Mock data for demonstration
- `src/utils/` - Utility functions
