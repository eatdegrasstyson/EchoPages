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

## Getting Started
### Prerequisites (In Progress)
Python 3.x

Required Libraries: pandas, os, json, nrclex, tensorflow/pytorch (for models)

### Installation
Clone the repository:

Bash
git clone https://github.com/eatdegrasstyson/EchoPages.git
Navigate to the project root and install dependencies.

## Roadmap
[ ] Integration of Music and Text engines for cross-modal sentiment mapping.

[ ] Refinement of the CRNN model for higher prediction accuracy.

[ ] Expansion of the dataset to include more diverse musical genres.