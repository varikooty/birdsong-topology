# British Birdsong Classification Project

A machine learning project for analyzing and classifying British bird songs using the [British Birdsong Dataset](https://www.kaggle.com/datasets/rtatman/british-birdsong-dataset/data).

## Project Overview

This project explores audio analysis and classification of British bird species through their songs. The work includes data exploration, feature extraction, visualization, and machine learning models for species classification.

## Setup Instructions

### 1. Clone this repository
```bash
git clone <your-repo-url>
cd birdsong_project
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
1. Go to [Kaggle Dataset](https://www.kaggle.com/datasets/rtatman/british-birdsong-dataset/data)
2. Download the dataset
3. Extract it to `data/birdsong/` directory

Expected structure:
```
birdsong_project/
├── data/
│   └── birdsong/
│       ├── species1/
│       │   ├── audio1.wav
│       │   └── audio2.wav
│       └── species2/
│           └── audio1.wav
├── notebooks/
│   └── 01_data_exploration.ipynb
├── src/
├── outputs/
└── requirements.txt
```

## Project Structure

- `data/` - Dataset storage (not included in repo)
- `notebooks/` - Jupyter notebooks for exploration and experimentation
- `src/` - Source code for models and utilities
- `outputs/` - Generated outputs (plots, models, results)
- `requirements.txt` - Python dependencies

## Getting Started

1. Start with the data exploration notebook:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

2. This notebook will help you:
   - Load and inspect the dataset
   - Visualize bird songs (waveforms, spectrograms)
   - Extract audio features
   - Understand the data distribution

## Approach

### Data Exploration
- Analyze audio properties (duration, sample rate, frequency content)
- Visualize spectrograms and mel-spectrograms
- Understand species distribution

### Feature Extraction
- MFCCs (Mel-Frequency Cepstral Coefficients)
- Spectral features (centroid, rolloff, bandwidth)
- Chroma features
- Zero-crossing rate

### Modeling (Next Steps)
- [To be implemented based on chosen direction]

## Tools & Libraries

- **Audio Processing**: librosa
- **Data Science**: numpy, pandas, scikit-learn
- **Visualization**: matplotlib, seaborn
- **Deep Learning**: PyTorch/TensorFlow
- **Notebook**: Jupyter

## Author

Vikram Varikooty  
B.S. Computer Science and Data Science  
University of Wisconsin-Madison, Class of 2027

## Acknowledgments

Dataset: British Birdsong Dataset by Rachael Tatman on Kaggle
