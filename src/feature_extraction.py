"""
Audio Feature Extraction Utilities
Extracts features from bird song audio files for machine learning
"""

import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class AudioFeatureExtractor:
    """Extract audio features for machine learning"""
    
    def __init__(self, sr: int = 22050, n_mfcc: int = 40, n_mels: int = 128):
        """
        Initialize feature extractor
        
        Args:
            sr: Sample rate for audio loading
            n_mfcc: Number of MFCCs to extract
            n_mels: Number of mel bands
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
    
    def extract_features(self, filepath: str, duration: Optional[float] = None) -> Dict:
        """
        Extract comprehensive audio features
        
        Args:
            filepath: Path to audio file
            duration: Maximum duration to load (None = entire file)
            
        Returns:
            Dictionary of features
        """
        try:
            # Load audio
            y, sr = librosa.load(filepath, sr=self.sr, duration=duration)
            
            features = {}
            
            # Time domain features
            features.update(self._extract_time_features(y, sr))
            
            # Frequency domain features
            features.update(self._extract_frequency_features(y, sr))
            
            # Mel features
            features.update(self._extract_mel_features(y, sr))
            
            # MFCC features
            features.update(self._extract_mfcc_features(y, sr))
            
            # Rhythm features
            features.update(self._extract_rhythm_features(y, sr))
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {filepath}: {e}")
            return None
    
    def _extract_time_features(self, y: np.ndarray, sr: int) -> Dict:
        """Extract time-domain features"""
        features = {}
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        # Energy
        energy = np.sum(y**2) / len(y)
        features['energy'] = float(energy)
        
        # RMS Energy
        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))
        
        return features
    
    def _extract_frequency_features(self, y: np.ndarray, sr: int) -> Dict:
        """Extract frequency-domain features"""
        features = {}
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
        features['spectral_centroid_std'] = float(np.std(spectral_centroid))
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
        features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features['spectral_contrast_mean'] = float(np.mean(spectral_contrast))
        features['spectral_contrast_std'] = float(np.std(spectral_contrast))
        
        # Spectral flatness
        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        features['spectral_flatness_mean'] = float(np.mean(spectral_flatness))
        features['spectral_flatness_std'] = float(np.std(spectral_flatness))
        
        return features
    
    def _extract_mel_features(self, y: np.ndarray, sr: int) -> Dict:
        """Extract mel-scale features"""
        features = {}
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        features['mel_mean'] = float(np.mean(mel_spec_db))
        features['mel_std'] = float(np.std(mel_spec_db))
        features['mel_max'] = float(np.max(mel_spec_db))
        features['mel_min'] = float(np.min(mel_spec_db))
        
        return features
    
    def _extract_mfcc_features(self, y: np.ndarray, sr: int) -> Dict:
        """Extract MFCC features"""
        features = {}
        
        # MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        
        # Statistics for each MFCC coefficient
        for i in range(self.n_mfcc):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfcc[i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfcc[i]))
        
        # Delta MFCCs (first derivative)
        mfcc_delta = librosa.feature.delta(mfcc)
        features['mfcc_delta_mean'] = float(np.mean(mfcc_delta))
        features['mfcc_delta_std'] = float(np.std(mfcc_delta))
        
        # Delta-delta MFCCs (second derivative)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        features['mfcc_delta2_mean'] = float(np.mean(mfcc_delta2))
        features['mfcc_delta2_std'] = float(np.std(mfcc_delta2))
        
        return features
    
    def _extract_rhythm_features(self, y: np.ndarray, sr: int) -> Dict:
        """Extract rhythm/tempo features"""
        features = {}
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = float(np.mean(chroma))
        features['chroma_std'] = float(np.std(chroma))
        
        return features


def process_dataset(data_path: str, output_csv: str, max_files: Optional[int] = None):
    """
    Process entire dataset and save features to CSV
    
    Args:
        data_path: Path to dataset directory
        output_csv: Output CSV file path
        max_files: Maximum number of files to process (None = all)
    """
    data_path = Path(data_path)
    extractor = AudioFeatureExtractor()
    
    # Find all audio files
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac']:
        audio_files.extend(data_path.rglob(ext))
    
    if max_files:
        audio_files = audio_files[:max_files]
    
    print(f"Processing {len(audio_files)} audio files...")
    
    # Extract features
    all_features = []
    for i, filepath in enumerate(audio_files):
        if i % 10 == 0:
            print(f"Processing file {i+1}/{len(audio_files)}...")
        
        # Extract species from directory structure
        species = filepath.parent.name
        
        # Extract features
        features = extractor.extract_features(str(filepath))
        
        if features:
            features['filepath'] = str(filepath)
            features['filename'] = filepath.name
            features['species'] = species
            all_features.append(features)
    
    # Save to CSV
    df = pd.DataFrame(all_features)
    df.to_csv(output_csv, index=False)
    print(f"\nFeatures saved to {output_csv}")
    print(f"Dataset shape: {df.shape}")
    print(f"Number of species: {df['species'].nunique()}")
    
    return df


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract features from bird song dataset')
    parser.add_argument('--data_path', type=str, default='../data/birdsong',
                       help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='../data/features.csv',
                       help='Output CSV file')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to process')
    
    args = parser.parse_args()
    
    df = process_dataset(args.data_path, args.output, args.max_files)
    print("\nFeature extraction complete!")
