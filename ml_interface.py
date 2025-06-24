#!/usr/bin/env python3
"""
Enhanced ML Interface for Gold Scalping - Adapted from BTC Strategy
Implements aggressive retraining and feature extraction for XAU/USD
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
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    ML_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ ML libraries not available. Install with: pip install scikit-learn")
    ML_AVAILABLE = False


class EnhancedMLSignal:
    """Enhanced ML signal with boosting capability"""
    def __init__(self, signal: str, confidence: float, reasoning: str, boost: int = 0):
        self.signal = signal        # 'buy', 'sell', 'hold'
        self.confidence = confidence # 0.0 to 1.0
        self.reasoning = reasoning
        self.boost = boost          # Signal boost for traditional logic
        self.timestamp = datetime.now()


class GoldFeatureExtractor:
    """Enhanced feature extractor adapted from BTC strategy"""
    
    def __init__(self, lookback_ticks: int = 30):  # Increased from BTC's 30
        self.lookback_ticks = lookback_ticks
        self.price_history = deque(maxlen=lookback_ticks)
        self.volume_history = deque(maxlen=lookback_ticks)
        self.timestamp_history = deque(maxlen=lookback_ticks)
        self.feature_names = []
        
        # Gold-specific thresholds (adjusted from BTC)
        self.momentum_thresholds = {
            'fast': 0.0015,      # 0.15% (5x BTC due to price ratio)
            'medium': 0.001,     # 0.10%
            'slow': 0.0005       # 0.05%
        }
    
    def add_tick(self, tick_data: Dict):
        """Add new tick data for feature extraction"""
        self.price_history.append(tick_data.get('price', 0))
        self.volume_history.append(tick_data.get('size', 1))
        self.timestamp_history.append(tick_data.get('timestamp', datetime.now()))
    
    def extract_enhanced_features(self) -> Dict:
        """Extract comprehensive features for ML (30+ features like BTC)"""
        
        if len(self.price_history) < 15:
            return {}
        
        prices = np.array(list(self.price_history))
        volumes = np.array(list(self.volume_history))
        
        features = {}
        
        # === PRICE FEATURES (BTC-style) ===
        features['current_price'] = prices[-1]
        features['price_change_3'] = self._calculate_price_change(3)
        features['price_change_5'] = self._calculate_price_change(5)
        features['price_change_10'] = self._calculate_price_change(10)
        features['price_change_15'] = self._calculate_price_change(15)
        
        # === MOMENTUM FEATURES (Gold-adjusted) ===
        features['momentum_fast'] = self._calculate_momentum(3)      # 3-tick momentum
        features['momentum_medium'] = self._calculate_momentum(5)    # 5-tick momentum
        features['momentum_slow'] = self._calculate_momentum(10)     # 10-tick momentum
        features['momentum_alignment'] = self._check_momentum_alignment()
        
        # === VOLATILITY FEATURES ===
        features['price_volatility_5'] = self._calculate_volatility(5)
        features['price_volatility_10'] = self._calculate_volatility(10)
        features['price_volatility_15'] = self._calculate_volatility(15)
        features['volatility_trend'] = self._calculate_volatility_trend()
        
        # === MOVING AVERAGE FEATURES ===
        features['sma_3'] = self._calculate_sma(3)
        features['sma_5'] = self._calculate_sma(5)
        features['sma_10'] = self._calculate_sma(10)
        features['sma_15'] = self._calculate_sma(15)
        features['price_above_sma3'] = 1 if prices[-1] > features['sma_3'] else 0
        features['price_above_sma5'] = 1 if prices[-1] > features['sma_5'] else 0
        features['price_above_sma10'] = 1 if prices[-1] > features['sma_10'] else 0
        features['sma3_above_sma5'] = 1 if features['sma_3'] > features['sma_5'] else 0
        features['sma5_above_sma10'] = 1 if features['sma_5'] > features['sma_10'] else 0
        
        # === TECHNICAL INDICATORS ===
        features['rsi_10'] = self._calculate_rsi(10)
        features['rsi_14'] = self._calculate_rsi(14)
        features['rsi_oversold'] = 1 if features['rsi_14'] < 30 else 0  # Gold levels
        features['rsi_overbought'] = 1 if features['rsi_14'] > 70 else 0
        
        # === VOLUME FEATURES ===
        features['volume_ratio_5'] = self._calculate_volume_ratio(5)
        features['volume_spike'] = 1 if self._detect_volume_spike() else 0
        features['volume_trend_5'] = self._calculate_volume_trend(5)
        features['avg_volume'] = np.mean(volumes)
        features['current_volume'] = volumes[-1]
        
        # === PATTERN RECOGNITION (Gold-adjusted) ===
        features['bullish_breakout'] = 1 if self._detect_bullish_breakout() else 0
        features['bearish_breakdown'] = 1 if self._detect_bearish_breakdown() else 0
        features['consolidation_time'] = self._calculate_consolidation_time()
        features['price_range'] = (np.max(prices) - np.min(prices)) / np.mean(prices)
        
        # === MARKET STATE FEATURES ===
        features['micro_trend'] = self._detect_micro_trend()
        features['trend_strength'] = self._calculate_trend_strength()
        features['market_regime'] = self._classify_market_regime()
        
        # Store feature names for consistency
        if not self.feature_names:
            self.feature_names = sorted(features.keys())
        
        return features
    
    def _calculate_price_change(self, periods: int) -> float:
        """Calculate price change over periods"""
        if len(self.price_history) < periods + 1:
            return 0.0
        
        current = self.price_history[-1]
        past = self.price_history[-(periods + 1)]
        
        return ((current - past) / past) * 100 if past != 0 else 0.0
    
    def _calculate_momentum(self, periods: int) -> float:
        """Calculate momentum (same as price change but as ratio)"""
        if len(self.price_history) < periods + 1:
            return 0.0
        
        current = self.price_history[-1]
        past = self.price_history[-(periods + 1)]
        
        return (current - past) / past if past != 0 else 0.0
    
    def _check_momentum_alignment(self) -> float:
        """Check if all momentum timeframes align (BTC feature)"""
        if len(self.price_history) < 11:
            return 0.0
        
        mom_fast = self._calculate_momentum(3)
        mom_medium = self._calculate_momentum(5) 
        mom_slow = self._calculate_momentum(10)
        
        # All positive or all negative = aligned
        if (mom_fast > 0 and mom_medium > 0 and mom_slow > 0):
            return 1.0  # Bullish alignment
        elif (mom_fast < 0 and mom_medium < 0 and mom_slow < 0):
            return -1.0  # Bearish alignment
        else:
            return 0.0  # No alignment
    
    def _calculate_volatility(self, periods: int) -> float:
        """Calculate rolling volatility"""
        if len(self.price_history) < periods:
            return 0.0
        
        prices = np.array(list(self.price_history)[-periods:])
        return np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0.0
    
    def _calculate_volatility_trend(self) -> float:
        """Calculate if volatility is increasing or decreasing"""
        if len(self.price_history) < 20:
            return 0.0
        
        recent_vol = self._calculate_volatility(10)
        past_vol = self._calculate_volatility_past(10, 10)  # 10 periods ago
        
        return (recent_vol - past_vol) / past_vol if past_vol > 0 else 0.0
    
    def _calculate_volatility_past(self, periods: int, offset: int) -> float:
        """Calculate volatility from offset periods ago"""
        if len(self.price_history) < periods + offset:
            return 0.0
        
        prices = np.array(list(self.price_history)[-(periods + offset):-offset])
        return np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0.0
    
    def _calculate_sma(self, periods: int) -> float:
        """Calculate Simple Moving Average"""
        if len(self.price_history) < periods:
            return 0.0
        
        prices = np.array(list(self.price_history)[-periods:])
        return np.mean(prices)
    
    def _calculate_rsi(self, periods: int = 14) -> float:
        """Calculate RSI with specified period"""
        if len(self.price_history) < periods + 1:
            return 50.0  # Neutral
        
        prices = np.array(list(self.price_history)[-periods-1:])
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_volume_ratio(self, periods: int) -> float:
        """Calculate current volume vs average"""
        if len(self.volume_history) < periods:
            return 1.0
        
        current_volume = self.volume_history[-1]
        avg_volume = np.mean(list(self.volume_history)[-periods:])
        
        return current_volume / avg_volume if avg_volume > 0 else 1.0
    
    def _detect_volume_spike(self) -> bool:
        """Detect volume spike (BTC-style)"""
        if len(self.volume_history) < 5:
            return False
        
        current_volume = self.volume_history[-1]
        avg_volume = np.mean(list(self.volume_history)[-5:])
        
        return current_volume > avg_volume * 1.3  # 30% above average
    
    def _calculate_volume_trend(self, periods: int) -> float:
        """Calculate volume trend"""
        if len(self.volume_history) < periods * 2:
            return 0.0
        
        recent_avg = np.mean(list(self.volume_history)[-periods:])
        past_avg = np.mean(list(self.volume_history)[-(periods*2):-periods])
        
        return (recent_avg - past_avg) / past_avg if past_avg > 0 else 0.0
    
    def _detect_bullish_breakout(self) -> bool:
        """Detect bullish breakout (gold-adjusted)"""
        if len(self.price_history) < 10:
            return False
        
        current_price = self.price_history[-1]
        recent_high = max(list(self.price_history)[-10:-1])
        
        # Gold breakout threshold (higher than BTC)
        return current_price > recent_high * 1.0025  # 0.25% breakout
    
    def _detect_bearish_breakdown(self) -> bool:
        """Detect bearish breakdown (gold-adjusted)"""
        if len(self.price_history) < 10:
            return False
        
        current_price = self.price_history[-1]
        recent_low = min(list(self.price_history)[-10:-1])
        
        return current_price < recent_low * 0.9975  # 0.25% breakdown
    
    def _calculate_consolidation_time(self) -> float:
        """Calculate how long price has been consolidating"""
        if len(self.price_history) < 10:
            return 0.0
        
        prices = np.array(list(self.price_history)[-10:])
        price_range = (np.max(prices) - np.min(prices)) / np.mean(prices)
        
        # Return inverse of range (higher = more consolidation)
        return 1.0 / (price_range + 0.001)
    
    def _detect_micro_trend(self) -> float:
        """Detect micro trend direction"""
        if len(self.price_history) < 5:
            return 0.0
        
        prices = np.array(list(self.price_history)[-5:])
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        
        # Normalize slope
        avg_price = np.mean(prices)
        normalized_slope = slope / avg_price if avg_price > 0 else 0.0
        
        return normalized_slope
    
    def _calculate_trend_strength(self) -> float:
        """Calculate overall trend strength"""
        if len(self.price_history) < 15:
            return 0.0
        
        prices = np.array(list(self.price_history)[-15:])
        
        # Linear regression slope
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)
        
        # R-squared to measure trend strength
        y_pred = slope * x + intercept
        ss_res = np.sum((prices - y_pred) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Return signed trend strength
        trend_direction = 1 if slope > 0 else -1
        return r_squared * trend_direction
    
    def _classify_market_regime(self) -> float:
        """Classify current market regime"""
        if len(self.price_history) < 15:
            return 0.0  # Unknown
        
        volatility = self._calculate_volatility(10)
        trend_strength = abs(self._calculate_trend_strength())
        
        if volatility > 0.005 and trend_strength > 0.7:
            return 3.0  # High volatility trending
        elif volatility > 0.005:
            return 2.0  # High volatility ranging
        elif trend_strength > 0.7:
            return 1.0  # Low volatility trending
        else:
            return 0.0  # Low volatility ranging


class EnhancedMLModel:
    """Enhanced ML model with aggressive retraining (BTC-style)"""
    
    def __init__(self, model_file: str = "gold_enhanced_ml.pkl"):
        self.model_file = model_file
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_trained = False
        self.model_version = 0
        
        # Training data storage
        self.training_features = []
        self.training_labels = []
        self.training_outcomes = []
        
        # Performance tracking
        self.predictions_made = 0
        self.correct_predictions = 0
        
        # Aggressive retraining (like BTC strategy)
        self.retrain_interval = 15  # Retrain every 15 samples (vs 50 for simple)
        
        # Load existing model if available
        self.load_model()
    
    def add_training_data(self, features: Dict, outcome: str, profit_loss: float):
        """Add training data with aggressive retraining"""
        
        if not features:
            return
        
        # Convert outcome to label (BTC method)
        if outcome == 'profitable' or profit_loss > 0:
            label = 1  # Profitable trade
        elif outcome == 'unprofitable' or profit_loss <= 0:
            label = 0  # Unprofitable trade
        else:
            return  # Skip unknown outcomes
        
        # Store training data
        self.training_features.append(features.copy())
        self.training_labels.append(label)
        self.training_outcomes.append(profit_loss)
        
        # Aggressive retraining (BTC-style)
        if len(self.training_features) >= self.retrain_interval and \
           len(self.training_features) % self.retrain_interval == 0:
            self.train_model()
    
    def train_model(self, min_samples: int = 15):  # Lower threshold than simple ML
        """Train ML model with aggressive settings"""
        
        if not ML_AVAILABLE:
            logging.warning("ML libraries not available for training")
            return False
        
        if len(self.training_features) < min_samples:
            logging.debug(f"Need {min_samples} samples to train. Have {len(self.training_features)}")
            return False
        
        try:
            # Prepare data
            X, y = self._prepare_training_data()
            
            if len(X) == 0:
                return False
            
            # Split data (if enough samples)
            if len(X) > 20:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=42
                )
            else:
                X_train, X_test, y_train, y_test = X, X, y, y  # Use all data for small sets
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Use Gradient Boosting like BTC strategy
            self.model = GradientBoostingClassifier(
                n_estimators=100,    # Same as BTC
                learning_rate=0.1,   # Same as BTC
                max_depth=4,         # Same as BTC
                random_state=42
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            self.model_version += 1
            self.save_model()
            
            logging.info(f"✅ Enhanced ML model trained! v{self.model_version} | Accuracy: {accuracy:.2f} | Samples: {len(X_train)}")
            return True
            
        except Exception as e:
            logging.error(f"Enhanced ML training error: {e}")
            return False
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training arrays with consistent features"""
        
        if not self.training_features:
            return np.array([]), np.array([])
        
        # Get feature names from first sample (ensure consistency)
        if not self.feature_names:
            self.feature_names = sorted(list(self.training_features[0].keys()))
        
        # Create feature matrix
        X = []
        y = []
        
        for features, label in zip(self.training_features, self.training_labels):
            # Create feature vector with consistent ordering
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
    
    def predict_enhanced(self, features: Dict) -> EnhancedMLSignal:
        """Make enhanced prediction with signal boosting (BTC-style)"""
        
        if not self.is_trained or not self.model:
            return EnhancedMLSignal('hold', 0.0, 'Model not trained', 0)
        
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
            
            # Enhanced signal generation (BTC-style)
            if prediction == 1 and confidence > 0.6:  # Profitable predicted
                signal = 'buy'
                boost = 3  # Strong boost to traditional signals
                reasoning = f'ML v{self.model_version}: Profitable trade predicted (conf: {confidence:.2f})'
                
            elif prediction == 0 and confidence > 0.6:  # Unprofitable predicted
                signal = 'hold'  # Avoid trades (vs 'sell' which would be shorting)
                boost = -2  # Negative boost (reduces traditional signals)
                reasoning = f'ML v{self.model_version}: Unprofitable trade avoided (conf: {confidence:.2f})'
                
            else:
                signal = 'hold'
                boost = 0
                reasoning = f'ML v{self.model_version}: Low confidence ({confidence:.2f})'
            
            self.predictions_made += 1
            
            return EnhancedMLSignal(signal, confidence, reasoning, boost)
            
        except Exception as e:
            logging.error(f"Enhanced ML prediction error: {e}")
            return EnhancedMLSignal('hold', 0.0, f'Prediction error: {e}', 0)
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from trained model"""
        
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
        """Save enhanced model to file"""
        
        if not self.is_trained:
            return
        
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_version': self.model_version,
                'training_samples': len(self.training_features),
                'predictions_made': self.predictions_made,
                'correct_predictions': self.correct_predictions,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            logging.info(f"✅ Enhanced ML model v{self.model_version} saved to {self.model_file}")
            
        except Exception as e:
            logging.error(f"Error saving enhanced model: {e}")
    
    def load_model(self):
        """Load enhanced model from file"""
        
        if not os.path.exists(self.model_file):
            return False
        
        try:
            with open(self.model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.model_version = model_data.get('model_version', 1)
            self.predictions_made = model_data.get('predictions_made', 0)
            self.correct_predictions = model_data.get('correct_predictions', 0)
            self.is_trained = True
            
            logging.info(f"✅ Enhanced ML model v{self.model_version} loaded from {self.model_file}")
            logging.info(f"   Training samples: {model_data.get('training_samples', 'unknown')}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error loading enhanced model: {e}")
            return False


class EnhancedMLInterface:
    """Enhanced ML interface with BTC-style integration"""
    
    def __init__(self, config: Dict = None):
        if config is None:
            config = ENHANCED_ML_CONFIG
        
        self.feature_extractor = GoldFeatureExtractor(
            lookback_ticks=config.get('lookback_ticks', 30)
        )
        self.ml_model = EnhancedMLModel(
            model_file=config.get('model_file', 'gold_enhanced_ml.pkl')
        )
        
        # Performance tracking
        self.predictions_made = 0
        self.correct_predictions = 0
        self.ml_accuracy = 0.0
        
        logging.info("✅ Enhanced ML interface initialized")
    
    def process_tick(self, tick_data: Dict) -> EnhancedMLSignal:
        """Process tick and return enhanced ML signal"""
        
        # Add tick to feature extractor
        self.feature_extractor.add_tick(tick_data)
        
        # Extract enhanced features
        features = self.feature_extractor.extract_enhanced_features()
        
        if not features:
            return EnhancedMLSignal('hold', 0.0, 'Insufficient data for features', 0)
        
        # Get enhanced ML prediction
        ml_signal = self.ml_model.predict_enhanced(features)
        
        if ml_signal.signal != 'hold':
            self.predictions_made += 1
        
        return ml_signal
    
    def record_trade_outcome(self, features: Dict, signal: str, profit_loss: float):
        """Record trade outcome for enhanced learning"""
        
        if not features:
            return
        
        # Determine outcome
        if signal == 'hold':
            outcome = 'no_trade'
        elif profit_loss > 0:
            outcome = 'profitable'
            self.correct_predictions += 1
        else:
            outcome = 'unprofitable'
        
        # Add to training data (aggressive retraining)
        self.ml_model.add_training_data(features, outcome, profit_loss)
        
        # Update accuracy
        if self.predictions_made > 0:
            self.ml_accuracy = (self.correct_predictions / self.predictions_made) * 100
        
        logging.debug(f"Enhanced ML outcome: {outcome} | P&L: ${profit_loss:.2f} | Accuracy: {self.ml_accuracy:.1f}%")
    
    def get_enhanced_ml_stats(self) -> Dict:
        """Get comprehensive ML statistics"""
        
        feature_importance = self.ml_model.get_feature_importance()
        
        return {
            'ml_available': ML_AVAILABLE,
            'model_trained': self.ml_model.is_trained,
            'model_version': self.ml_model.model_version,
            'training_samples': len(self.ml_model.training_features),
            'predictions_made': self.predictions_made,
            'correct_predictions': self.correct_predictions,
            'accuracy': self.ml_accuracy,
            'retrain_interval': self.ml_model.retrain_interval,
            'feature_count': len(self.feature_extractor.feature_names),
            'top_features': dict(list(feature_importance.items())[:10])  # Top 10 features
        }
    
    def force_retrain(self):
        """Force model retraining (BTC-style)"""
        
        if ML_AVAILABLE:
            success = self.ml_model.train_model(min_samples=10)  # Lower threshold
            if success:
                logging.info(f"✅ Enhanced ML model v{self.ml_model.model_version} retrained")
            else:
                logging.warning("⚠️ Enhanced ML retraining failed")
        else:
            logging.warning("⚠️ ML libraries not available")
    
    def get_ml_signal_integration(self, traditional_signal, ml_signal: EnhancedMLSignal) -> Dict:
        """Integrate ML signal with traditional signal (BTC-style)"""
        
        if ml_signal.signal == 'hold' or ml_signal.boost == 0:
            return {
                'final_signal': traditional_signal.signal_type,
                'final_confidence': traditional_signal.confidence,
                'ml_enhancement': 'none',
                'reasoning': traditional_signal.reasoning
            }
        
        # Apply ML boost to traditional scoring
        if hasattr(traditional_signal, 'bullish_score') and hasattr(traditional_signal, 'bearish_score'):
            enhanced_bullish = traditional_signal.bullish_score
            enhanced_bearish = traditional_signal.bearish_score
            
            if ml_signal.signal == 'buy' and ml_signal.boost > 0:
                enhanced_bullish += ml_signal.boost
            elif ml_signal.boost < 0:  # ML suggests avoiding
                enhanced_bullish = max(0, enhanced_bullish + ml_signal.boost)
                enhanced_bearish = max(0, enhanced_bearish + ml_signal.boost)
            
            # Determine final signal
            if enhanced_bullish > enhanced_bearish and enhanced_bullish >= 2:
                final_signal = 'buy'
                final_confidence = min(0.95, 0.3 + (enhanced_bullish / 8))
            elif enhanced_bearish > enhanced_bullish and enhanced_bearish >= 2:
                final_signal = 'sell'
                final_confidence = min(0.95, 0.3 + (enhanced_bearish / 8))
            else:
                final_signal = 'hold'
                final_confidence = traditional_signal.confidence
            
            return {
                'final_signal': final_signal,
                'final_confidence': final_confidence,
                'ml_enhancement': f'boosted by {ml_signal.boost}',
                'reasoning': f"{traditional_signal.reasoning} + {ml_signal.reasoning}",
                'scores': f"B{enhanced_bullish}/B{enhanced_bearish} (ML boosted)"
            }
        
        # Fallback to simple integration
        return {
            'final_signal': traditional_signal.signal_type,
            'final_confidence': min(traditional_signal.confidence + 0.1, 0.95),
            'ml_enhancement': 'confidence boost',
            'reasoning': f"{traditional_signal.reasoning} + {ml_signal.reasoning}"
        }


# Enhanced Configuration
ENHANCED_ML_CONFIG = {
    'lookback_ticks': 30,
    'model_file': 'gold_enhanced_scalping_ml.pkl',
    'min_confidence': 0.45,
    'retrain_interval': 15,  # Aggressive like BTC
    'feature_count_target': 35
}

# Compatibility wrapper
class SimpleMLInterface(EnhancedMLInterface):
    """Wrapper for compatibility with existing main.py"""
    
    def get_ml_stats(self):
        """Compatibility method"""
        return self.get_enhanced_ml_stats()


if __name__ == "__main__":
    # Test enhanced ML interface
    print("🤖 Testing Enhanced Gold ML Interface...")
    
    if not ML_AVAILABLE:
        print("❌ ML libraries not available - install scikit-learn")
        exit(1)
    
    # Create enhanced ML interface
    ml_interface = EnhancedMLInterface(ENHANCED_ML_CONFIG)
    
    # Simulate realistic gold tick data
    import random
    base_price = 2000.0
    
    print("🧪 Simulating gold ticks for ML training...")
    
    for i in range(50):  # More ticks for better testing
        # Generate realistic price movement
        price_change = random.uniform(-0.8, 0.8)
        base_price += price_change
        
        tick_data = {
            'price': round(base_price, 2),
            'size': random.randint(1, 100),
            'timestamp': datetime.now()
        }
        
        # Process tick
        signal = ml_interface.process_tick(tick_data)
        
        if signal.signal != 'hold':
            print(f"ML Signal: {signal.signal.upper()} @ ${base_price:.2f}")
            print(f"  Confidence: {signal.confidence:.2f} | Boost: {signal.boost}")
            print(f"  Reasoning: {signal.reasoning}")
            
            # Simulate trade outcome
            outcome_pnl = random.uniform(-1.0, 2.0)  # Slightly positive bias
            features = ml_interface.feature_extractor.extract_enhanced_features()
            ml_interface.record_trade_outcome(features, signal.signal, outcome_pnl)
    
    # Print enhanced ML stats
    stats = ml_interface.get_enhanced_ml_stats()
    print(f"\n🤖 Enhanced ML Stats:")
    print(f"   Model Version: v{stats['model_version']}")
    print(f"   Training Samples: {stats['training_samples']}")
    print(f"   Predictions Made: {stats['predictions_made']}")
    print(f"   Accuracy: {stats['accuracy']:.1f}%")
    print(f"   Feature Count: {stats['feature_count']}")
    print(f"   Retrain Interval: {stats['retrain_interval']}")
    
    if stats['top_features']:
        print(f"   Top Features: {list(stats['top_features'].keys())[:5]}")
    
    # Force retrain
    print("\n🔄 Testing aggressive retraining...")
    ml_interface.force_retrain()
    
    print("✅ Enhanced ML Interface test completed")