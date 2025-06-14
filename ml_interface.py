#!/usr/bin/env python3
"""
Simple ML Interface for XAU/USD Trading Bot
"""

import logging
import pickle
import os
import numpy as np
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional, Tuple

# Try to import ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    ML_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ ML libraries not available. Install with: pip install scikit-learn")
    ML_AVAILABLE = False


class SimpleMLSignal:
    """Simple ML signal structure"""
    def __init__(self, signal: str, confidence: float, reasoning: str):
        self.signal = signal        # 'buy', 'sell', 'hold'
        self.confidence = confidence # 0.0 to 1.0
        self.reasoning = reasoning


class SimpleFeatureExtractor:
    """Extract features from tick data for ML"""
    
    def __init__(self, lookback_ticks: int = 20):
        self.lookback_ticks = lookback_ticks
        self.price_history = deque(maxlen=lookback_ticks)
        self.volume_history = deque(maxlen=lookback_ticks)
    
    def add_tick(self, tick_data: Dict):
        """Add new tick data"""
        self.price_history.append(tick_data.get('price', 0))
        self.volume_history.append(tick_data.get('size', 1))
    
    def extract_features(self) -> Dict:
        """Extract simple features for ML"""
        
        if len(self.price_history) < 10:
            return {}
        
        prices = np.array(list(self.price_history))
        volumes = np.array(list(self.volume_history))
        
        features = {}
        
        # Price features
        features['current_price'] = prices[-1]
        features['price_change_5'] = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        features['price_change_10'] = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
        features['price_volatility'] = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0
        features['price_trend'] = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
        
        # Moving averages
        features['sma_5'] = np.mean(prices[-5:])
        features['sma_10'] = np.mean(prices[-10:])
        features['price_above_sma5'] = 1 if prices[-1] > features['sma_5'] else 0
        features['price_above_sma10'] = 1 if prices[-1] > features['sma_10'] else 0
        
        # Volume features
        features['avg_volume'] = np.mean(volumes)
        features['volume_trend'] = (np.mean(volumes[-5:]) - np.mean(volumes[-10:-5])) if len(volumes) >= 10 else 0
        features['current_volume'] = volumes[-1]
        features['volume_ratio'] = volumes[-1] / np.mean(volumes) if np.mean(volumes) > 0 else 1
        
        # Technical indicators (simplified)
        features['momentum'] = (prices[-1] - prices[-3]) / prices[-3] if len(prices) >= 3 and prices[-3] > 0 else 0
        features['rsi_approx'] = self._calculate_simple_rsi(prices)
        
        return features
    
    def _calculate_simple_rsi(self, prices: np.ndarray, period: int = 10) -> float:
        """Calculate simplified RSI"""
        if len(prices) < period + 1:
            return 50.0  # Neutral
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class SimpleMLModel:
    """Simple ML model for trading predictions"""
    
    def __init__(self, model_file: str = "simple_trading_model.pkl"):
        self.model_file = model_file
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_trained = False
        
        # Training data storage
        self.training_features = []
        self.training_labels = []
        
        # Load existing model if available
        self.load_model()
    
    def add_training_data(self, features: Dict, outcome: str, profit_loss: float):
        """Add data for model training"""
        
        if not features:
            return
        
        # Convert outcome to label
        if outcome == 'profitable':
            label = 1  # Buy signal worked
        elif outcome == 'unprofitable':
            label = 0  # Signal didn't work
        else:
            return  # Skip no-trade outcomes
        
        # Store training data
        self.training_features.append(features.copy())
        self.training_labels.append(label)
        
        # Auto-train every 50 samples
        if len(self.training_features) >= 50 and len(self.training_features) % 25 == 0:
            self.train_model()
    
    def train_model(self, min_samples: int = 30):
        """Train the ML model"""
        
        if not ML_AVAILABLE:
            logging.warning("ML libraries not available for training")
            return False
        
        if len(self.training_features) < min_samples:
            logging.info(f"Need {min_samples} samples to train. Have {len(self.training_features)}")
            return False
        
        try:
            # Prepare data
            X, y = self._prepare_training_data()
            
            if len(X) == 0:
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=42
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            self.save_model()
            
            logging.info(f"✅ ML model trained! Accuracy: {accuracy:.2f} | Samples: {len(X_train)}")
            return True
            
        except Exception as e:
            logging.error(f"ML training error: {e}")
            return False
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training arrays"""
        
        if not self.training_features:
            return np.array([]), np.array([])
        
        # Get feature names from first sample
        self.feature_names = sorted(list(self.training_features[0].keys()))
        
        # Create feature matrix
        X = []
        y = []
        
        for features, label in zip(self.training_features, self.training_labels):
            # Create feature vector
            feature_vector = []
            for feature_name in self.feature_names:
                value = features.get(feature_name, 0)
                # Handle invalid values
                if np.isnan(value) or np.isinf(value):
                    value = 0
                feature_vector.append(value)
            
            X.append(feature_vector)
            y.append(label)
        
        return np.array(X), np.array(y)
    
    def predict(self, features: Dict) -> SimpleMLSignal:
        """Make prediction using trained model"""
        
        if not self.is_trained or not self.model:
            return SimpleMLSignal('hold', 0.0, 'Model not trained')
        
        try:
            # Prepare feature vector
            feature_vector = []
            for feature_name in self.feature_names:
                value = features.get(feature_name, 0)
                if np.isnan(value) or np.isinf(value):
                    value = 0
                feature_vector.append(value)
            
            X = np.array([feature_vector])
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            confidence = max(probabilities)
            
            # Convert prediction to signal
            if prediction == 1 and confidence > 0.6:
                signal = 'buy'
                reasoning = f'ML predicts profitable trade (confidence: {confidence:.2f})'
            elif prediction == 0 and confidence > 0.6:
                signal = 'sell'  # Or could be 'hold' depending on strategy
                reasoning = f'ML predicts unprofitable trade (confidence: {confidence:.2f})'
            else:
                signal = 'hold'
                reasoning = f'ML confidence too low: {confidence:.2f}'
            
            return SimpleMLSignal(signal, confidence, reasoning)
            
        except Exception as e:
            logging.error(f"ML prediction error: {e}")
            return SimpleMLSignal('hold', 0.0, f'Prediction error: {e}')
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from model"""
        
        if not self.is_trained or not self.model:
            return {}
        
        try:
            importances = self.model.feature_importances_
            importance_dict = {}
            
            for i, feature_name in enumerate(self.feature_names):
                importance_dict[feature_name] = importances[i]
            
            # Sort by importance
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_features)
            
        except Exception as e:
            logging.error(f"Error getting feature importance: {e}")
            return {}
    
    def save_model(self):
        """Save model to file"""
        
        if not self.is_trained:
            return
        
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'training_samples': len(self.training_features),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            logging.info(f"✅ ML model saved to {self.model_file}")
            
        except Exception as e:
            logging.error(f"Error saving model: {e}")
    
    def load_model(self):
        """Load model from file"""
        
        if not os.path.exists(self.model_file):
            return False
        
        try:
            with open(self.model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.is_trained = True
            
            logging.info(f"✅ ML model loaded from {self.model_file}")
            logging.info(f"   Training samples: {model_data.get('training_samples', 'unknown')}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return False


class SimpleMLInterface:
    """Main ML interface for the trading bot"""
    
    def __init__(self, config: Dict = None):
        if config is None:
            config = {}
        
        self.feature_extractor = SimpleFeatureExtractor(
            lookback_ticks=config.get('lookback_ticks', 20)
        )
        self.ml_model = SimpleMLModel(
            model_file=config.get('model_file', 'xauusd_ml_model.pkl')
        )
        
        # Performance tracking
        self.predictions_made = 0
        self.correct_predictions = 0
        
        logging.info("✅ Simple ML interface initialized")
    
    def process_tick(self, tick_data: Dict) -> SimpleMLSignal:
        """Process tick and return ML signal"""
        
        # Add tick to feature extractor
        self.feature_extractor.add_tick(tick_data)
        
        # Extract features
        features = self.feature_extractor.extract_features()
        
        if not features:
            return SimpleMLSignal('hold', 0.0, 'Insufficient data for features')
        
        # Get ML prediction
        ml_signal = self.ml_model.predict(features)
        
        if ml_signal.signal != 'hold':
            self.predictions_made += 1
        
        return ml_signal
    
    def record_trade_outcome(self, features: Dict, signal: str, profit_loss: float):
        """Record trade outcome for model learning"""
        
        if not features:
            return
        
        # Determine outcome
        if signal == 'hold':
            outcome = 'no_trade'
        elif profit_loss > 0:
            outcome = 'profitable'
            if self.predictions_made > 0:
                self.correct_predictions += 1
        else:
            outcome = 'unprofitable'
        
        # Add to training data
        self.ml_model.add_training_data(features, outcome, profit_loss)
        
        logging.debug(f"ML outcome recorded: {outcome} | P&L: ${profit_loss:.2f}")
    
    def get_ml_stats(self) -> Dict:
        """Get ML statistics"""
        
        accuracy = (self.correct_predictions / self.predictions_made * 100) if self.predictions_made > 0 else 0
        feature_importance = self.ml_model.get_feature_importance()
        
        return {
            'ml_available': ML_AVAILABLE,
            'model_trained': self.ml_model.is_trained,
            'training_samples': len(self.ml_model.training_features),
            'predictions_made': self.predictions_made,
            'correct_predictions': self.correct_predictions,
            'accuracy': accuracy,
            'top_features': dict(list(feature_importance.items())[:5])  # Top 5 features
        }
    
    def force_retrain(self):
        """Force model retraining"""
        
        if ML_AVAILABLE:
            success = self.ml_model.train_model(min_samples=20)
            if success:
                logging.info("✅ ML model retrained")
            else:
                logging.warning("⚠️ ML retraining failed")
        else:
            logging.warning("⚠️ ML libraries not available")


# Configuration for ML Interface
ML_CONFIG = {
    'lookback_ticks': 20,
    'model_file': 'xauusd_simple_ml.pkl',
    'min_confidence': 0.65,
    'retrain_interval': 50
}


if __name__ == "__main__":
    # Test ML interface
    print("🤖 Testing Simple ML Interface...")
    
    if not ML_AVAILABLE:
        print("❌ ML libraries not available - install scikit-learn")
        exit(1)
    
    # Create ML interface
    ml_interface = SimpleMLInterface(ML_CONFIG)
    
    # Simulate some tick data
    import random
    base_price = 2000.0
    
    for i in range(30):
        # Generate realistic tick
        price_change = random.uniform(-0.5, 0.5)
        base_price += price_change
        
        tick_data = {
            'price': base_price,
            'size': random.randint(1, 50),
            'timestamp': datetime.now()
        }
        
        # Process tick
        signal = ml_interface.process_tick(tick_data)
        
        if signal.signal != 'hold':
            print(f"Signal: {signal.signal} | Confidence: {signal.confidence:.2f} | {signal.reasoning}")
            
            # Simulate trade outcome
            outcome_pnl = random.uniform(-0.5, 1.0)  # Random P&L
            features = ml_interface.feature_extractor.extract_features()
            ml_interface.record_trade_outcome(features, signal.signal, outcome_pnl)
    
    # Print ML stats
    stats = ml_interface.get_ml_stats()
    print(f"\n🤖 ML Stats: {stats}")
    
    # Force retrain
    ml_interface.force_retrain()
    
    print("✅ ML Interface test completed")