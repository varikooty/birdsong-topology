"""
Bird Song Classification Model
Trains and evaluates models for classifying bird species from audio features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


class BirdSongClassifier:
    """Train and evaluate bird song classification models"""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize classifier
        
        Args:
            model_type: Type of model ('random_forest', 'svm', 'gradient_boosting')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Initialize model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            self.model = SVC(kernel='rbf', random_state=42, probability=True)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels from dataframe
        
        Args:
            df: DataFrame with features and species column
            
        Returns:
            X: Feature matrix
            y: Label array
        """
        # Drop non-feature columns
        feature_cols = [col for col in df.columns 
                       if col not in ['filepath', 'filename', 'species']]
        
        X = df[feature_cols].values
        y = df['species'].values
        
        return X, y
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train_scaled, y_train_encoded)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train_encoded, cv=5)
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test set
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Encode and scale
        y_test_encoded = self.label_encoder.transform(y_test)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_encoded, y_pred)
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test_encoded, y_pred, 
                                   target_names=self.label_encoder.classes_))
        
        # Confusion matrix
        cm = confusion_matrix(y_test_encoded, y_pred)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'true_labels': y_test_encoded
        }
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = None):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            save_path: Path to save plot (optional)
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'Confusion Matrix - {self.model_type}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_names: list, top_n: int = 20, 
                               save_path: str = None):
        """
        Plot feature importance (for tree-based models)
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to display
            save_path: Path to save plot (optional)
        """
        if not hasattr(self.model, 'feature_importances_'):
            print("Feature importance not available for this model type")
            return
        
        # Get feature importances
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top {top_n} Feature Importances')
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n), [feature_names[i] for i in indices], 
                  rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, filepath: str):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load trained model"""
        model_data = joblib.load(filepath)
        classifier = cls(model_type=model_data['model_type'])
        classifier.model = model_data['model']
        classifier.scaler = model_data['scaler']
        classifier.label_encoder = model_data['label_encoder']
        print(f"Model loaded from {filepath}")
        return classifier


def main():
    """Main training pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train bird song classifier')
    parser.add_argument('--features', type=str, default='../data/features.csv',
                       help='Path to features CSV')
    parser.add_argument('--model_type', type=str, default='random_forest',
                       choices=['random_forest', 'svm', 'gradient_boosting'],
                       help='Type of model to train')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size')
    parser.add_argument('--output_dir', type=str, default='../outputs',
                       help='Output directory for plots and models')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    print("Loading features...")
    df = pd.read_csv(args.features)
    print(f"Dataset shape: {df.shape}")
    print(f"Number of species: {df['species'].nunique()}")
    print(f"Species distribution:\n{df['species'].value_counts()}")
    
    # Initialize classifier
    classifier = BirdSongClassifier(model_type=args.model_type)
    
    # Prepare data
    X, y = classifier.prepare_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train model
    classifier.train(X_train, y_train)
    
    # Evaluate
    results = classifier.evaluate(X_test, y_test)
    
    # Plot confusion matrix
    cm_path = output_dir / f'confusion_matrix_{args.model_type}.png'
    classifier.plot_confusion_matrix(results['confusion_matrix'], str(cm_path))
    
    # Plot feature importance (if available)
    feature_cols = [col for col in df.columns 
                   if col not in ['filepath', 'filename', 'species']]
    
    fi_path = output_dir / f'feature_importance_{args.model_type}.png'
    classifier.plot_feature_importance(feature_cols, save_path=str(fi_path))
    
    # Save model
    model_path = output_dir / f'model_{args.model_type}.pkl'
    classifier.save_model(str(model_path))
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
