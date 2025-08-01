"""
Custom Neural Networks for content analysis and classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModel
import logging
import asyncio
import pickle
import json
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


class ContentDataset(Dataset):
    """Dataset class for content classification"""
    
    def __init__(self, texts: List[str], labels: List[int] = None, tokenizer=None, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        if self.tokenizer:
            # Tokenize for transformer models
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            item = {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            }
        else:
            # Simple text for traditional models
            item = {'text': text}
        
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item


class ContentClassifierNet(nn.Module):
    """Neural network for content classification"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, dropout_rate: float = 0.3):
        super(ContentClassifierNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        logits = self.network(x)
        return logits
    
    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return self.softmax(logits)


class QualityScorer(nn.Module):
    """Neural network for content quality scoring"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64, 32]):
        super(QualityScorer, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        # Output layer for regression (quality score 0-1)
        layers.extend([
            nn.Linear(prev_size, 1),
            nn.Sigmoid()
        ])
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class DuplicateDetector(nn.Module):
    """Neural network for duplicate content detection"""
    
    def __init__(self, embedding_size: int = 256):
        super(DuplicateDetector, self).__init__()
        
        self.embedding_size = embedding_size
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Similarity network
        self.similarity = nn.Sequential(
            nn.Linear(64, 32),  # Concatenated embeddings
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x1, x2):
        # Encode both inputs
        emb1 = self.encoder(x1)
        emb2 = self.encoder(x2)
        
        # Concatenate embeddings
        combined = torch.cat([emb1, emb2], dim=1)
        
        # Calculate similarity
        similarity = self.similarity(combined)
        
        return similarity, emb1, emb2


class ContentClassifier:
    """Main class for content classification and analysis"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize models
        self.classifier_model = None
        self.quality_model = None
        self.duplicate_model = None
        self.vectorizer = None
        
        # Model parameters
        self.hidden_sizes = [256, 128, 64]
        self.num_classes = 10  # Predefined categories
        self.max_features = 5000
        
        # Load or initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize or load pre-trained models"""
        try:
            # Try to load existing models
            self._load_models()
        except:
            # Create new models with default weights
            logger.info("Creating new neural network models")
            self._create_default_models()
    
    def _create_default_models(self):
        """Create default models with random weights"""
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        # Create models
        self.classifier_model = ContentClassifierNet(
            input_size=self.max_features,
            hidden_sizes=self.hidden_sizes,
            num_classes=self.num_classes
        )
        
        self.quality_model = QualityScorer(
            input_size=self.max_features
        )
        
        self.duplicate_model = DuplicateDetector(
            embedding_size=self.max_features
        )
        
        # Set to evaluation mode
        self.classifier_model.eval()
        self.quality_model.eval()
        self.duplicate_model.eval()
    
    def _load_models(self):
        """Load pre-trained models from disk"""
        classifier_path = self.model_dir / "classifier.pth"
        quality_path = self.model_dir / "quality_scorer.pth"
        duplicate_path = self.model_dir / "duplicate_detector.pth"
        vectorizer_path = self.model_dir / "vectorizer.pkl"
        
        if all(path.exists() for path in [classifier_path, quality_path, duplicate_path, vectorizer_path]):
            # Load vectorizer
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # Create and load models
            self.classifier_model = ContentClassifierNet(
                input_size=self.max_features,
                hidden_sizes=self.hidden_sizes,
                num_classes=self.num_classes
            )
            self.classifier_model.load_state_dict(torch.load(classifier_path, map_location='cpu'))
            
            self.quality_model = QualityScorer(input_size=self.max_features)
            self.quality_model.load_state_dict(torch.load(quality_path, map_location='cpu'))
            
            self.duplicate_model = DuplicateDetector(embedding_size=self.max_features)
            self.duplicate_model.load_state_dict(torch.load(duplicate_path, map_location='cpu'))
            
            # Set to evaluation mode
            self.classifier_model.eval()
            self.quality_model.eval()
            self.duplicate_model.eval()
            
            logger.info("Successfully loaded pre-trained models")
        else:
            raise FileNotFoundError("Model files not found")
    
    def _save_models(self):
        """Save models to disk"""
        try:
            # Save models
            torch.save(self.classifier_model.state_dict(), self.model_dir / "classifier.pth")
            torch.save(self.quality_model.state_dict(), self.model_dir / "quality_scorer.pth")
            torch.save(self.duplicate_model.state_dict(), self.model_dir / "duplicate_detector.pth")
            
            # Save vectorizer
            with open(self.model_dir / "vectorizer.pkl", 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            logger.info("Models saved successfully")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    async def classify_content(self, text: str) -> Dict[str, Any]:
        """Classify content into predefined categories"""
        try:
            # Preprocess text
            text_features = self._preprocess_text(text)
            
            if text_features is None:
                return {"error": "Failed to process text"}
            
            # Run classification
            with torch.no_grad():
                text_tensor = torch.FloatTensor(text_features).unsqueeze(0)
                probabilities = self.classifier_model.predict_proba(text_tensor)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities).item()
            
            # Map to category names
            categories = [
                'technology', 'business', 'science', 'health', 'education',
                'entertainment', 'sports', 'politics', 'travel', 'general'
            ]
            
            predicted_category = categories[predicted_class] if predicted_class < len(categories) else 'general'
            
            # Get all category probabilities
            category_probs = {}
            for i, category in enumerate(categories):
                if i < probabilities.shape[1]:
                    category_probs[category] = float(probabilities[0][i].item())
            
            return {
                'predicted_category': predicted_category,
                'confidence': float(confidence),
                'category_probabilities': category_probs,
                'prediction_timestamp': asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Content classification failed: {e}")
            return {"error": str(e)}
    
    async def score_quality(self, text: str) -> Dict[str, Any]:
        """Score content quality using neural network"""
        try:
            # Preprocess text
            text_features = self._preprocess_text(text)
            
            if text_features is None:
                return {"error": "Failed to process text"}
            
            # Run quality scoring
            with torch.no_grad():
                text_tensor = torch.FloatTensor(text_features).unsqueeze(0)
                quality_score = self.quality_model(text_tensor).item()
            
            # Categorize quality
            if quality_score >= 0.8:
                quality_level = "Excellent"
            elif quality_score >= 0.6:
                quality_level = "Good"
            elif quality_score >= 0.4:
                quality_level = "Average"
            elif quality_score >= 0.2:
                quality_level = "Poor"
            else:
                quality_level = "Very Poor"
            
            return {
                'quality_score': float(quality_score),
                'quality_level': quality_level,
                'normalized_score': float(quality_score * 100),  # 0-100 scale
                'scoring_timestamp': asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Quality scoring failed: {e}")
            return {"error": str(e)}
    
    async def detect_duplicates(self, text1: str, text2: str) -> Dict[str, Any]:
        """Detect if two texts are duplicates"""
        try:
            # Preprocess both texts
            features1 = self._preprocess_text(text1)
            features2 = self._preprocess_text(text2)
            
            if features1 is None or features2 is None:
                return {"error": "Failed to process texts"}
            
            # Run duplicate detection
            with torch.no_grad():
                tensor1 = torch.FloatTensor(features1).unsqueeze(0)
                tensor2 = torch.FloatTensor(features2).unsqueeze(0)
                
                similarity, emb1, emb2 = self.duplicate_model(tensor1, tensor2)
                similarity_score = similarity.item()
            
            # Determine if duplicate
            threshold = 0.8
            is_duplicate = similarity_score >= threshold
            
            # Calculate additional similarity metrics
            cosine_sim = self._cosine_similarity(features1, features2)
            jaccard_sim = self._jaccard_similarity(text1, text2)
            
            return {
                'is_duplicate': is_duplicate,
                'neural_similarity': float(similarity_score),
                'cosine_similarity': float(cosine_sim),
                'jaccard_similarity': float(jaccard_sim),
                'threshold': threshold,
                'confidence': float(abs(similarity_score - threshold)),
                'detection_timestamp': asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Duplicate detection failed: {e}")
            return {"error": str(e)}
    
    async def batch_classify(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify multiple texts in batch"""
        results = []
        
        try:
            # Preprocess all texts
            all_features = []
            valid_indices = []
            
            for i, text in enumerate(texts):
                features = self._preprocess_text(text)
                if features is not None:
                    all_features.append(features)
                    valid_indices.append(i)
                else:
                    results.append({"error": f"Failed to process text {i}"})
            
            if not all_features:
                return results
            
            # Run batch classification
            with torch.no_grad():
                text_tensor = torch.FloatTensor(np.array(all_features))
                probabilities = self.classifier_model.predict_proba(text_tensor)
                predictions = torch.argmax(probabilities, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]
            
            # Process results
            categories = [
                'technology', 'business', 'science', 'health', 'education',
                'entertainment', 'sports', 'politics', 'travel', 'general'
            ]
            
            batch_results = []
            for i, (pred, conf, probs) in enumerate(zip(predictions, confidences, probabilities)):
                predicted_category = categories[pred.item()] if pred.item() < len(categories) else 'general'
                
                category_probs = {}
                for j, category in enumerate(categories):
                    if j < probs.shape[0]:
                        category_probs[category] = float(probs[j].item())
                
                batch_results.append({
                    'predicted_category': predicted_category,
                    'confidence': float(conf.item()),
                    'category_probabilities': category_probs
                })
            
            # Merge results maintaining original order
            final_results = [None] * len(texts)
            for i, result in enumerate(batch_results):
                final_results[valid_indices[i]] = result
            
            # Fill in failed entries
            for i, result in enumerate(final_results):
                if result is None:
                    final_results[i] = {"error": f"Failed to process text {i}"}
            
            return final_results
            
        except Exception as e:
            logger.error(f"Batch classification failed: {e}")
            return [{"error": str(e)} for _ in texts]
    
    def _preprocess_text(self, text: str) -> Optional[np.ndarray]:
        """Preprocess text for neural network input"""
        try:
            if not text or len(text.strip()) < 5:
                return None
            
            # Ensure vectorizer is fitted or use dummy data
            if not hasattr(self.vectorizer, 'vocabulary_'):
                # Fit on dummy data to create vocabulary
                dummy_texts = [
                    "technology software computer programming",
                    "business market finance economy",
                    "science research experiment analysis",
                    "health medical doctor treatment",
                    "education school learning student"
                ]
                self.vectorizer.fit(dummy_texts)
            
            # Transform text
            features = self.vectorizer.transform([text]).toarray()
            return features[0]
            
        except Exception as e:
            logger.error(f"Text preprocessing failed: {e}")
            return None
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception:
            return 0.0
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts"""
        try:
            # Tokenize texts
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            if union == 0:
                return 0.0
            
            return intersection / union
            
        except Exception:
            return 0.0
    
    async def train_classifier(
        self,
        texts: List[str],
        labels: List[int],
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ) -> Dict[str, Any]:
        """Train the classification model on provided data"""
        
        try:
            # Preprocess texts
            self.vectorizer.fit(texts)
            features = []
            valid_indices = []
            
            for i, text in enumerate(texts):
                processed = self._preprocess_text(text)
                if processed is not None:
                    features.append(processed)
                    valid_indices.append(i)
            
            # Filter labels for valid texts
            valid_labels = [labels[i] for i in valid_indices]
            
            if len(features) < 2:
                return {"error": "Insufficient training data"}
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                features, valid_labels, test_size=0.2, random_state=42
            )
            
            # Create data loaders
            train_dataset = ContentDataset(X_train, y_train)
            val_dataset = ContentDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Initialize model and optimizer
            self.classifier_model = ContentClassifierNet(
                input_size=self.max_features,
                hidden_sizes=self.hidden_sizes,
                num_classes=self.num_classes
            )
            
            optimizer = optim.Adam(self.classifier_model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            train_losses = []
            val_accuracies = []
            
            for epoch in range(epochs):
                # Training
                self.classifier_model.train()
                epoch_loss = 0.0
                
                for batch in train_loader:
                    optimizer.zero_grad()
                    
                    # Assuming batch contains preprocessed features
                    if isinstance(batch['text'], list):
                        # Convert list of features to tensor
                        inputs = torch.FloatTensor([self._preprocess_text(t) for t in batch['text']])
                    else:
                        inputs = batch['text']
                    
                    labels = batch['label']
                    
                    outputs = self.classifier_model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                # Validation
                self.classifier_model.eval()
                val_predictions = []
                val_true_labels = []
                
                with torch.no_grad():
                    for batch in val_loader:
                        if isinstance(batch['text'], list):
                            inputs = torch.FloatTensor([self._preprocess_text(t) for t in batch['text']])
                        else:
                            inputs = batch['text']
                        
                        labels = batch['label']
                        outputs = self.classifier_model(inputs)
                        predictions = torch.argmax(outputs, dim=1)
                        
                        val_predictions.extend(predictions.cpu().numpy())
                        val_true_labels.extend(labels.cpu().numpy())
                
                # Calculate metrics
                val_accuracy = accuracy_score(val_true_labels, val_predictions)
                avg_loss = epoch_loss / len(train_loader)
                
                train_losses.append(avg_loss)
                val_accuracies.append(val_accuracy)
                
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
            # Save trained model
            self._save_models()
            
            return {
                'success': True,
                'final_loss': train_losses[-1],
                'final_accuracy': val_accuracies[-1],
                'training_history': {
                    'losses': train_losses,
                    'accuracies': val_accuracies
                },
                'model_saved': True
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {"error": str(e)}
    
    async def evaluate_model(self, test_texts: List[str], test_labels: List[int]) -> Dict[str, Any]:
        """Evaluate model performance on test data"""
        try:
            # Get predictions
            predictions = await self.batch_classify(test_texts)
            
            # Extract predicted classes
            predicted_classes = []
            for pred in predictions:
                if 'error' not in pred:
                    categories = [
                        'technology', 'business', 'science', 'health', 'education',
                        'entertainment', 'sports', 'politics', 'travel', 'general'
                    ]
                    predicted_category = pred['predicted_category']
                    predicted_class = categories.index(predicted_category) if predicted_category in categories else 9
                    predicted_classes.append(predicted_class)
                else:
                    predicted_classes.append(9)  # Default to 'general'
            
            # Calculate metrics
            accuracy = accuracy_score(test_labels, predicted_classes)
            
            # Detailed classification report
            categories = [
                'technology', 'business', 'science', 'health', 'education',
                'entertainment', 'sports', 'politics', 'travel', 'general'
            ]
            
            report = classification_report(
                test_labels,
                predicted_classes,
                target_names=categories,
                output_dict=True,
                zero_division=0
            )
            
            return {
                'accuracy': float(accuracy),
                'classification_report': report,
                'num_test_samples': len(test_texts),
                'evaluation_timestamp': asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {"error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current models"""
        return {
            'classifier': {
                'type': 'ContentClassifierNet',
                'input_size': self.max_features,
                'hidden_sizes': self.hidden_sizes,
                'num_classes': self.num_classes,
                'parameters': sum(p.numel() for p in self.classifier_model.parameters()) if self.classifier_model else 0
            },
            'quality_scorer': {
                'type': 'QualityScorer',
                'input_size': self.max_features,
                'parameters': sum(p.numel() for p in self.quality_model.parameters()) if self.quality_model else 0
            },
            'duplicate_detector': {
                'type': 'DuplicateDetector',
                'embedding_size': self.max_features,
                'parameters': sum(p.numel() for p in self.duplicate_model.parameters()) if self.duplicate_model else 0
            },
            'vectorizer': {
                'type': 'TfidfVectorizer',
                'max_features': self.max_features,
                'vocabulary_size': len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, 'vocabulary_') else 0
            }
        }