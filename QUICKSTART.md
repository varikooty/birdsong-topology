# Quick Start Guide

## Step-by-Step Instructions to Get Running

### 1. Download the Dataset

1. Go to https://www.kaggle.com/datasets/rtatman/british-birdsong-dataset/data
2. Download the dataset (you'll need a Kaggle account)
3. Extract the zip file
4. Move the extracted folder to `data/birdsong/` in this project

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 3. Run Data Exploration

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/01_data_exploration.ipynb
# Run all cells to explore the data
```

### 4. Extract Features

```bash
# Extract features from all audio files
python src/feature_extraction.py --data_path data/birdsong --output data/features.csv

# Or for testing, process just 100 files:
python src/feature_extraction.py --data_path data/birdsong --output data/features.csv --max_files 100
```

This will create a CSV file with extracted audio features.

### 5. Train a Classifier

```bash
# Train a Random Forest classifier
python src/classifier.py --features data/features.csv --model_type random_forest

# Or try different models:
python src/classifier.py --features data/features.csv --model_type svm
python src/classifier.py --features data/features.csv --model_type gradient_boosting
```

Results will be saved in the `outputs/` directory:
- Confusion matrix plot
- Feature importance plot
- Trained model (.pkl file)

### 6. Next Steps

After getting the baseline working, consider:

1. **Deep Learning Approach**
   - Use spectrograms as images
   - Train a CNN (ResNet, EfficientNet)
   - Try a recurrent model for temporal patterns

2. **Advanced Visualization**
   - t-SNE or UMAP of audio features
   - Interactive spectrograms
   - Audio playback in notebooks

3. **Topological Data Analysis** (aligns with the research group!)
   - Apply persistent homology to spectrograms
   - Use topological features for classification
   - Show understanding of geometric/topological methods

4. **Generative Models**
   - VAE or GAN for bird song synthesis
   - Audio style transfer

## Troubleshooting

**"No module named 'librosa'"**
- Make sure you activated the virtual environment
- Run `pip install -r requirements.txt`

**"No such file or directory: data/birdsong"**
- Download the dataset from Kaggle first
- Make sure it's extracted to the correct location

**Audio files not loading**
- Check that ffmpeg is installed on your system
- On Mac: `brew install ffmpeg`
- On Ubuntu: `sudo apt-get install ffmpeg`

## Tips for Success

1. **Start Simple**: Get the baseline classification working first
2. **Document Well**: Comment your code and explain your thinking
3. **Show Creativity**: Try something unique that aligns with the lab's research
4. **Be Clear**: Write a good README explaining what you did and why
5. **Visualize**: Include plots and visualizations to show your results

## Questions?

If you get stuck, feel free to:
- Check the documentation for librosa: https://librosa.org/
- Look at scikit-learn examples: https://scikit-learn.org/
- Review the original Kaggle dataset for ideas

Good luck!
