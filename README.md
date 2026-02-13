<<<<<<< HEAD
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
=======
# Topological Analysis of British Bird Songs

A novel approach to bird song classification using **Topological Data Analysis (TDA)** and persistent homology, applied to the British Birdsong Dataset.

## Overview

This project explores bird song classification by extracting topological features from audio spectrograms using persistent homology. Unlike traditional audio features (MFCCs, spectral features), topological features capture the geometric and structural patterns in bird vocalizations, providing complementary information about the shape and connectivity of sound patterns.

## Motivation

Traditional audio classification relies on statistical features that may miss important structural information. Topological Data Analysis offers a mathematically rigorous framework for capturing:
- **Connected components (H0)**: Discrete sound events and syllable structure
- **Loops and holes (H1)**: Temporal patterns, harmonics, and cyclic structures in vocalizations

This approach aligns with modern geometric deep learning methods and provides interpretable features grounded in algebraic topology.

## Dataset

**British Birdsong Dataset** from Kaggle
- 264 audio recordings (.flac format)
- 88 bird species from the UK
- Metadata includes species names, genus, location, and recording details

## Methodology

### 1. Audio Preprocessing
- Load audio files at 22.05 kHz sampling rate
- Generate mel-spectrograms (128 mel bands)
- Convert to decibel scale for better dynamic range

### 2. Topological Feature Extraction
- Sample high-intensity points from spectrograms (top 25th percentile)
- Compute persistent homology using Ripser library
- Extract persistence diagrams for H0 and H1 homology groups
- Calculate statistical features from persistence:
  - Number of topological features
  - Mean, std, max, sum of persistence values
  - Birth and death time statistics

### 3. Classification
- Compare topological vs. traditional audio features
- Random Forest classifier with 3-fold cross-validation
- Analyze feature importance to identify discriminative topological signatures

## Key Findings

### Topological Features Captured

The analysis reveals that different bird species have distinct topological signatures:

- **H0 (Connected Components)**: Captures the number and distribution of discrete sound events
  - Different species show varying patterns of syllable clustering
  - Persistence values indicate the strength and duration of distinct sound events

- **H1 (Loops/Cycles)**: Captures harmonic and temporal structure
  - Reveals cyclic patterns in bird vocalizations
  - Identifies species-specific harmonic relationships

### Classification Performance

Analyzed 18 samples from 3 species (Great Tit, Tree Sparrow, Marsh Tit):

| Method | Accuracy | Notes |
|--------|----------|-------|
| Topological Features | 22% ± 21% | Captures geometric structure |
| Traditional Features | 33% ± 0% | Standard audio features |
| Combined Features | 33% ± 14% | Complementary information |

**Note**: Limited sample size (6 per species) restricts statistical power. Results demonstrate proof-of-concept for topological feature extraction.

### Most Important Features

Top discriminative topological features:
1. H0 persistence statistics (mean, sum)
2. H1 feature counts
3. Birth/death time distributions

## Visualizations

The project generates three key visualizations:

1. **Persistence Diagrams**: Shows topological signatures for each species
2. **Feature Importance**: Identifies most discriminative topological features
3. **Comprehensive Analysis**: Combines spectrograms, waveforms, and topology

All visualizations are saved in `outputs/` directory.

## Project Structure

```
birdsong_project/
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Initial data analysis
│   └── 02_topological_analysis.ipynb  # TDA implementation
├── outputs/
│   ├── persistence_diagrams.png
│   ├── feature_importance.png
│   └── comprehensive_analysis.png
├── data/
│   └── birdsong/                      # Audio files (not in repo)
├── requirements.txt
└── README.md
```

## Requirements

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
librosa>=0.9.0
scikit-learn>=1.0.0
ripser>=0.6.0
persim>=0.3.0
scikit-tda>=1.0.0
```

## Installation & Usage

```bash
# Clone repository
git clone https://github.com/yourusername/birdsong-topology.git
cd birdsong-topology

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
# Place in data/birdsong/

# Run analysis
jupyter notebook notebooks/02_topological_analysis.ipynb
```

## Future Directions

1. **Scale to full dataset**: Analyze all 264 recordings across 88 species
2. **Deep topological features**: Combine with neural networks (topological layers)
3. **Temporal dynamics**: Apply sliding window TDA to capture temporal evolution
4. **Multi-scale analysis**: Explore different resolution levels in persistence
5. **Comparative topology**: Study topological differences between bird families/genera

## Technical Approach

### Why Topological Data Analysis?

Traditional audio features treat spectrograms as statistical distributions. TDA treats them as geometric objects, capturing:
- **Shape**: Overall structure of vocalizations
- **Connectivity**: How sound components relate
- **Persistence**: Which features are robust vs. noise

This geometric perspective is particularly relevant for:
- Bird songs with complex temporal structure
- Species with distinctive harmonic patterns
- Vocalizations with repeating motifs

### Computational Complexity

- Point cloud sampling: O(n) where n = spectrogram points
- Persistent homology: O(n³) worst case, optimized by Ripser
- Feature extraction: O(k) where k = number of topological features

Runtime: ~2-3 seconds per audio file on standard hardware

## References

- **Ripser**: Fast computation of Vietoris-Rips persistence barcodes
- **Persistent Homology**: Edelsbrunner & Harer, "Computational Topology"
- **TDA in Audio**: Chung & Day, "Topological approaches to deep learning"
>>>>>>> 0918a99069a98578bbeffcf0344f26de83ee89c8

## Author

Vikram Varikooty  
B.S. Computer Science and Data Science  
University of Wisconsin-Madison, Class of 2027

## Acknowledgments

<<<<<<< HEAD
Dataset: British Birdsong Dataset by Rachael Tatman on Kaggle
=======
- British Birdsong Dataset by Rachael Tatman (Kaggle)
- Ripser and scikit-tda development teams
- UW-Madison Biomedical Engineering Department

---

*This project demonstrates the application of topological data analysis to audio classification, showcasing how geometric and algebraic topology can extract meaningful features from complex temporal signals.*
>>>>>>> 0918a99069a98578bbeffcf0344f26de83ee89c8
