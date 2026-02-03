EchoPages
EchoPages is a multi-modal analysis platform designed to bridge the gap between music and text through emotional intelligence. By leveraging audio features and sentiment analysis, the project aims to categorize and predict emotional responses to various media.

## Project Structure
The repository is organized into several specialized engines:

### Music Engine
The MusicEngine is the core component for processing audio data and metadata.

SpotifyToMeta: Contains scripts like CombineDatasets.py and ComputeMetadata.py to aggregate music data from sources like MuVi, Trompa-Mer, and Music-Mouv.

SpotifyToSpectrogram: Handles the scraping and conversion of audio files into spectrograms (.npy) for audio analysis and training.

Model: Includes train_CRNN.py and PredictEmotionFromSong.py for training and running Convolutional Recurrent Neural Networks on music data.

DataSets: Contains the raw and processed datasets used for training and validation, including FINALDATASET.csv (the primary dataset used for model training.)

Temp: Holds some our spectrogram and audio files for viewing and experimenting (may be removed)

### Text Engine
The TextEngine focuses on extracting, analyzing, and modeling emotional content from written text.

Preprocessing: Handles text cleaning, normalization, and tokenization. Removes noise such as punctuation, stop words, and formatting artifacts to prepare text for analysis.

Sentiment & Emotion Analysis: Uses lexicon-based tools (e.g., NRC Emotion Lexicon via nrclex) and/or learned models to map text to emotional categories such as joy, sadness, anger, fear, trust, and anticipation.

Feature Extraction: Converts text into numerical representations (e.g., emotion scores, sentiment polarity, frequency-based features).

Modeling (Planned / In Progress):

## Getting Started
### Prerequisites (In Progress)
Python 3.x

Required Libraries: pandas, os, json, nrclex, tensorflow/pytorch (for models)

### Installation
Clone the repository:

Bash
git clone https://github.com/eatdegrasstyson/EchoPages.git
Navigate to the project root and install dependencies.

