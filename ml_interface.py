#!/usr/bin/env python3
"""
Machine Learning Interface for Tick Trading Bot
Implements pattern recognition and adaptive trading strategies
"""

import logging
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import deque
import json

# Try to import ML libraries, fallback gracefully if not available
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.feature_selection import SelectKBest, f_classif
    ML_AVAILABLE = True
except ImportError:
    logging.warning("⚠️  Scikit-learn not available. Install with: pip install scikit-learn pandas")
    ML_AVAILABLE = False

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    logging.warning("⚠️  TA-Lib not available. Some indicators will use simplified versions")
    TALIB_AVAILABLE = False


class TickFeatureEngineer:
    """Feature engineering for tick-level data"""
    
    def __init__(self, lookback_ticks: int = 100):
        self.lookback_ticks = lookback_ticks
        self.tick_buffer = deque(maxlen=lookback_ticks)
        
    def add_tick(self, tick: Dict):
        """Add a new tick to the buffer"""
        self.tick_buffer.append(tick)
    
    def extract_features(self) -> Dict:
        """Extract comprehensive features from tick data"""
        if len(self.tick_buffer) < 20:
            return {}
        
        ticks = list(self.tick_buffer)
        prices = np.array([tick['price'] for tick in ticks])
        sizes = np.array([tick.get('size', 1) for tick in ticks])
        timestamps = [tick['timestamp'] for tick in ticks]
        
        features = {}
        
        # Basic price features
        features.update(self._extract_price_features(prices))
        
        # Volume/size features
        features.update(self._extract_volume_features(sizes))
        
        # Time-based features
        features.update(self._extract_time_features(timestamps))
        
        # Technical indicators
        features.update(self._extract_technical_features(prices))
        
        # Tick velocity and acceleration
        features.update(self._extract_velocity_features(timestamps, prices))
        
        # Spread and liquidity features
        features.update(self._extract_spread_features(ticks))
        
        # Market microstructure features
        features.update(self._extract_microstructure_features(ticks))
        
        return features
    
    def _extract_price_features(self, prices: np.ndarray) -> Dict:
        """Extract price-based features"""
        if len(prices) < 5:
            return {}
        
        # Basic statistics
        current_price = prices[-1]
        features = {
            'price_mean_10': np.mean(prices[-10:]),
            'price_mean_20': np.mean(prices[-20:]),
            'price_std_10': np.std(prices[-10:]),
            'price_std_20': np.std(prices[-20:]),
            'price_range_10': np.max(prices[-10:]) - np.min(prices[-10:]),
            'price_range_20': np.max(prices[-20:]) - np.min(prices[-20:]),
        }
        
        # Price momentum
        if len(prices) >= 10:
            features['momentum_5'] = (current_price - prices[-5]) / prices[-5]
            features['momentum_10'] = (current_price - prices[-10]) / prices[-10]
        
        # Price percentile position
        if len(prices) >= 20:
            features['price_percentile_20'] = (np.sum(prices[-20:] <= current_price) / 20) * 100
        
        # Rate of change
        if len(prices) >= 3:
            roc_short = (prices[-1] - prices[-3]) / prices[-3]
            features['roc_3_ticks'] = roc_short
        
        # Support and resistance levels
        features.update(self._find_support_resistance(prices))
        
        return features
    
    def _extract_volume_features(self, sizes: np.ndarray) -> Dict:
        """Extract volume/size based features"""
        if len(sizes) < 5:
            return {}
        
        return {
            'volume_mean_10': np.mean(sizes[-10:]),
            'volume_std_10': np.std(sizes[-10:]),
            'volume_ratio_current': sizes[-1] / np.mean(sizes[-10:]) if np.mean(sizes[-10:]) > 0 else 1,
            'volume_trend_5': np.mean(sizes[-5:]) - np.mean(sizes[-10:-5]) if len(sizes) >= 10 else 0,
            'large_tick_ratio': np.sum(sizes[-10:] > np.mean(sizes[-20:])) / 10 if len(sizes) >= 20 else 0.5
        }
    
    def _extract_time_features(self, timestamps: List) -> Dict:
        """Extract time-based features"""
        if len(timestamps) < 5:
            return {}
        
        # Convert to time differences
        time_diffs = []
        for i in range(1, len(timestamps)):
            if hasattr(timestamps[i], 'timestamp'):
                diff = (timestamps[i].timestamp() - timestamps[i-1].timestamp()) * 1000  # milliseconds
            else:
                # Assume datetime objects
                diff = (timestamps[i] - timestamps[i-1]).total_seconds() * 1000
            time_diffs.append(diff)
        
        if not time_diffs:
            return {}
        
        return {
            'avg_tick_interval': np.mean(time_diffs[-10:]),
            'tick_interval_std': np.std(time_diffs[-10:]),
            'tick_acceleration': np.mean(time_diffs[-5:]) - np.mean(time_diffs[-10:-5]) if len(time_diffs) >= 10 else 0
        }
    
    def _extract_technical_features(self, prices: np.ndarray) -> Dict:
        """Extract technical analysis features"""
        features = {}
        
        if len(prices) < 20:
            return features
        
        # Simple moving averages
        sma_5 = np.mean(prices[-5:])
        sma_10 = np.mean(prices[-10:])
        sma_20 = np.mean(prices[-20:])
        
        features.update({
            'sma_5': sma_5,
            'sma_10': sma_10,
            'sma_20': sma_20,
            'price_vs_sma5': (prices[-1] - sma_5) / sma_5,
            'price_vs_sma10': (prices[-1] - sma_10) / sma_10,
            'sma_cross_signal': 1 if sma_5 > sma_10 else -1
        })
        
        # RSI approximation
        if len(prices) >= 14:
            features['rsi_approx'] = self._calculate_rsi_approx(prices, period=14)
        
        # Bollinger Bands approximation
        if len(prices) >= 20:
            bb_features = self._calculate_bollinger_bands(prices, period=20)
            features.update(bb_features)
        
        # MACD approximation
        if len(prices) >= 26:
            macd_features = self._calculate_macd_approx(prices)
            features.update(macd_features)
        
        return features
    
    def _extract_velocity_features(self, timestamps: List, prices: np.ndarray) -> Dict:
        """Extract velocity and acceleration features"""
        if len(timestamps) < 10 or len(prices) < 10:
            return {}
        
        # Calculate price velocity (price change per unit time)
        recent_times = timestamps[-10:]
        recent_prices = prices[-10:]
        
        try:
            time_spans = []
            price_changes = []
            
            for i in range(1, len(recent_times)):
                if hasattr(recent_times[i], 'timestamp'):
                    time_diff = recent_times[i].timestamp() - recent_times[i-1].timestamp()
                else:
                    time_diff = (recent_times[i] - recent_times[i-1]).total_seconds()
                
                if time_diff > 0:
                    time_spans.append(time_diff)
                    price_changes.append(recent_prices[i] - recent_prices[i-1])
            
            if time_spans and price_changes:
                velocities = [pc / ts for pc, ts in zip(price_changes, time_spans)]
                
                return {
                    'price_velocity_mean': np.mean(velocities),
                    'price_velocity_std': np.std(velocities),
                    'price_acceleration': np.mean(np.diff(velocities)) if len(velocities) > 1 else 0
                }
        except Exception as e:
            logging.warning(f"Error calculating velocity features: {e}")
        
        return {}
    
    def _extract_spread_features(self, ticks: List) -> Dict:
        """Extract bid-ask spread features"""
        spreads = []
        bid_ask_ratios = []
        
        for tick in ticks[-10:]:
            if 'bid' in tick and 'ask' in tick and tick['bid'] > 0 and tick['ask'] > 0:
                spread = tick['ask'] - tick['bid']
                spreads.append(spread)
                bid_ask_ratios.append(tick['bid'] / tick['ask'])
        
        if not spreads:
            return {}
        
        return {
            'avg_spread': np.mean(spreads),
            'spread_volatility': np.std(spreads),
            'current_spread': spreads[-1] if spreads else 0,
            'spread_percentile': (np.sum(np.array(spreads) <= spreads[-1]) / len(spreads)) * 100 if len(spreads) > 1 else 50
        }
    
    def _extract_microstructure_features(self, ticks: List) -> Dict:
        """Extract market microstructure features"""
        features = {}
        
        # Trade direction analysis (using Lee-Ready algorithm approximation)
        buy_volume = 0
        sell_volume = 0
        
        for tick in ticks[-10:]:
            if 'bid' in tick and 'ask' in tick:
                mid_price = (tick['bid'] + tick['ask']) / 2
                if tick['price'] > mid_price:
                    buy_volume += tick.get('size', 1)
                elif tick['price'] < mid_price:
                    sell_volume += tick.get('size', 1)
        
        total_volume = buy_volume + sell_volume
        if total_volume > 0:
            features['buy_sell_ratio'] = buy_volume / total_volume
            features['order_flow_imbalance'] = (buy_volume - sell_volume) / total_volume
        
        # Price impact estimation
        large_trades = [tick for tick in ticks[-10:] if tick.get('size', 1) > np.mean([t.get('size', 1) for t in ticks[-20:]])]
        features['large_trade_frequency'] = len(large_trades) / 10
        
        return features
    
    def _calculate_rsi_approx(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate approximate RSI"""
        if TALIB_AVAILABLE:
            return talib.RSI(prices, timeperiod=period)[-1]
        
        # Simple RSI approximation
        if len(prices) < period + 1:
            return 50
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20) -> Dict:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return {}
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        current_price = prices[-1]
        
        return {
            'bb_upper': upper_band,
            'bb_lower': lower_band,
            'bb_position': (current_price - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0.5,
            'bb_squeeze': std / sma if sma != 0 else 0
        }
    
    def _calculate_macd_approx(self, prices: np.ndarray) -> Dict:
        """Calculate approximate MACD"""
        if len(prices) < 26:
            return {}
        
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        macd_line = ema_12 - ema_26
        
        return {
            'macd': macd_line,
            'macd_signal': 1 if macd_line > 0 else -1
        }
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _find_support_resistance(self, prices: np.ndarray) -> Dict:
        """Find support and resistance levels"""
        if len(prices) < 20:
            return {}
        
        recent_prices = prices[-20:]
        current_price = prices[-1]
        
        # Simple support/resistance using local minima/maxima
        local_maxima = []
        local_minima = []
        
        for i in range(2, len(recent_prices) - 2):
            # Local maxima
            if (recent_prices[i] > recent_prices[i-1] and 
                recent_prices[i] > recent_prices[i+1] and
                recent_prices[i] > recent_prices[i-2] and 
                recent_prices[i] > recent_prices[i+2]):
                local_maxima.append(recent_prices[i])
            
            # Local minima
            if (recent_prices[i] < recent_prices[i-1] and 
                recent_prices[i] < recent_prices[i+1] and
                recent_prices[i] < recent_prices[i-2] and 
                recent_prices[i] < recent_prices[i+2]):
                local_minima.append(recent_prices[i])
        
        resistance_level = np.mean(local_maxima) if local_maxima else current_price
        support_level = np.mean(local_minima) if local_minima else current_price
        
        return {
            'resistance_level': resistance_level,
            'support_level': support_level,
            'distance_to_resistance': (resistance_level - current_price) / current_price if current_price > 0 else 0,
            'distance_to_support': (current_price - support_level) / current_price if current_price > 0 else 0
        }


class TradingPatternRecognizer:
    """Advanced pattern recognition for tick trading"""
    
    def __init__(self):
        self.known_patterns = {
            'momentum_burst': self._detect_momentum_burst,
            'reversal_signal': self._detect_reversal_signal,
            'breakout_pattern': self._detect_breakout_pattern,
            'scalping_opportunity': self._detect_scalping_opportunity,
            'volatility_spike': self._detect_volatility_spike
        }
        
    def detect_patterns(self, features: Dict) -> Dict[str, float]:
        """Detect trading patterns and return confidence scores"""
        pattern_scores = {}
        
        for pattern_name, detector_func in self.known_patterns.items():
            try:
                score = detector_func(features)
                pattern_scores[pattern_name] = score
            except Exception as e:
                logging.warning(f"Error detecting pattern {pattern_name}: {e}")
                pattern_scores[pattern_name] = 0.0
        
        return pattern_scores
    
    def _detect_momentum_burst(self, features: Dict) -> float:
        """Detect momentum burst patterns"""
        score = 0.0
        
        # High velocity with consistent direction
        if 'price_velocity_mean' in features and abs(features['price_velocity_mean']) > 0.1:
            score += 0.3
        
        # Volume confirmation
        if 'volume_ratio_current' in features and features['volume_ratio_current'] > 1.5:
            score += 0.3
        
        # Price acceleration
        if 'price_acceleration' in features and abs(features['price_acceleration']) > 0.05:
            score += 0.4
        
        return min(score, 1.0)
    
    def _detect_reversal_signal(self, features: Dict) -> float:
        """Detect potential reversal patterns"""
        score = 0.0
        
        # RSI extremes
        if 'rsi_approx' in features:
            rsi = features['rsi_approx']
            if rsi > 80 or rsi < 20:
                score += 0.4
        
        # Bollinger Band extremes
        if 'bb_position' in features:
            bb_pos = features['bb_position']
            if bb_pos > 0.95 or bb_pos < 0.05:
                score += 0.3
        
        # Volume divergence
        if 'volume_trend_5' in features and 'momentum_5' in features:
            vol_trend = features['volume_trend_5']
            momentum = features['momentum_5']
            if (momentum > 0 and vol_trend < 0) or (momentum < 0 and vol_trend > 0):
                score += 0.3
        
        return min(score, 1.0)
    
    def _detect_breakout_pattern(self, features: Dict) -> float:
        """Detect breakout patterns"""
        score = 0.0
        
        # Breaking support/resistance
        if 'distance_to_resistance' in features and abs(features['distance_to_resistance']) < 0.001:
            score += 0.4
        if 'distance_to_support' in features and abs(features['distance_to_support']) < 0.001:
            score += 0.4
        
        # Volume confirmation
        if 'large_trade_frequency' in features and features['large_trade_frequency'] > 0.3:
            score += 0.3
        
        # Volatility expansion
        if 'price_std_10' in features and 'price_std_20' in features:
            if features['price_std_10'] > features['price_std_20'] * 1.2:
                score += 0.3
        
        return min(score, 1.0)
    
    def _detect_scalping_opportunity(self, features: Dict) -> float:
        """Detect good scalping conditions"""
        score = 0.0
        
        # Tight spreads
        if 'current_spread' in features and features['current_spread'] < 0.1:
            score += 0.3
        
        # High tick velocity
        if 'tick_velocity_mean' in features and features['tick_velocity_mean'] > 5:
            score += 0.2
        
        # Moderate volatility
        if 'price_volatility' in features:
            vol = features['price_volatility']
            if 0.05 <= vol <= 0.15:  # Sweet spot for scalping
                score += 0.3
        
        # Good liquidity
        if 'avg_tick_size' in features and features['avg_tick_size'] > 10:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_volatility_spike(self, features: Dict) -> float:
        """Detect volatility spikes"""
        score = 0.0
        
        # Sudden volatility increase
        if 'price_std_10' in features and 'price_std_20' in features:
            if features['price_std_10'] > features['price_std_20'] * 1.5:
                score += 0.5
        
        # Large price movements
        if 'price_range_10' in features and 'price_range_20' in features:
            if features['price_range_10'] > features['price_range_20'] * 0.8:
                score += 0.3
        
        # Accelerating tick intervals
        if 'tick_acceleration' in features and features['tick_acceleration'] < -10:
            score += 0.2
        
        return min(score, 1.0)


class MLTradingModel:
    """Machine Learning model for trading decisions"""
    
    def __init__(self, model_file: str = "tick_trading_model.pkl"):
        self.model_file = model_file
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.feature_names = []
        self.training_data = []
        self.is_trained = False
        
        # Load existing model if available
        self.load_model()
    
    def collect_training_data(self, features: Dict, trade_outcome: str, profit_loss: float):
        """Collect data for model training"""
        if not features:
            return
        
        # Create label: 1 for profitable, 0 for unprofitable, 2 for no trade
        if trade_outcome == 'no_trade':
            label = 2
        elif profit_loss > 0:
            label = 1
        else:
            label = 0
        
        # Store training sample
        training_sample = {
            'features': features.copy(),
            'label': label,
            'profit_loss': profit_loss,
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_data.append(training_sample)
        
        # Auto-train every 100 samples
        if len(self.training_data) % 100 == 0:
            logging.info(f"Auto-training model with {len(self.training_data)} samples")
            self.train_model()
    
    def train_model(self, min_samples: int = 50):
        """Train the ML model"""
        if not ML_AVAILABLE:
            logging.warning("ML libraries not available for training")
            return False
        
        if len(self.training_data) < min_samples:
            logging.info(f"Need at least {min_samples} samples to train. Currently have {len(self.training_data)}")
            return False
        
        try:
            # Prepare training data
            X, y = self._prepare_training_data()
            
            if len(X) == 0:
                logging.warning("No valid training data available")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Feature scaling
            self.scaler = RobustScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Feature selection
            self.feature_selector = SelectKBest(f_classif, k=min(20, X_train.shape[1]))
            X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = self.feature_selector.transform(X_test_scaled)
            
            # Train model
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            self.model.fit(X_train_selected, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_selected)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train_selected, y_train, cv=3)
            
            logging.info(f"✅ Model trained successfully!")
            logging.info(f"   Accuracy: {accuracy:.3f}")
            logging.info(f"   CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            logging.info(f"   Training samples: {len(X_train)}")
            
            self.is_trained = True
            self.save_model()
            
            return True
            
        except Exception as e:
            logging.error(f"Error training model: {e}")
            return False
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data arrays"""
        if not self.training_data:
            return np.array([]), np.array([])
        
        # Get all unique feature names
        all_features = set()
        for sample in self.training_data:
            all_features.update(sample['features'].keys())
        
        self.feature_names = sorted(list(all_features))
        
        # Create feature matrix
        X = []
        y = []
        
        for sample in self.training_data:
            # Create feature vector
            feature_vector = []
            for feature_name in self.feature_names:
                value = sample['features'].get(feature_name, 0)
                # Handle nan values
                if np.isnan(value) or np.isinf(value):
                    value = 0
                feature_vector.append(value)
            
            X.append(feature_vector)
            y.append(sample['label'])
        
        return np.array(X), np.array(y)
    
    def predict_trade_signal(self, features: Dict) -> Dict:
        """Predict trading signal using the trained model"""
        if not self.is_trained or not self.model:
            return {'signal': 'hold', 'confidence': 0.0, 'probabilities': []}
        
        try:
            # Prepare feature vector
            feature_vector = []
            for feature_name in self.feature_names:
                value = features.get(feature_name, 0)
                if np.isnan(value) or np.isinf(value):
                    value = 0
                feature_vector.append(value)
            
            X = np.array([feature_vector])
            
            # Scale and select features
            X_scaled = self.scaler.transform(X)
            X_selected = self.feature_selector.transform(X_scaled)
            
            # Make prediction
            prediction = self.model.predict(X_selected)[0]
            probabilities = self.model.predict_proba(X_selected)[0]
            
            # Convert prediction to signal
            signal_map = {0: 'sell', 1: 'buy', 2: 'hold'}
            signal = signal_map.get(prediction, 'hold')
            
            confidence = max(probabilities)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'probabilities': probabilities.tolist(),
                'prediction': int(prediction)
            }
            
        except Exception as e:
            logging.error(f"Error making prediction: {e}")
            return {'signal': 'hold', 'confidence': 0.0, 'probabilities': []}
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from the trained model"""
        if not self.is_trained or not self.model:
            return {}
        
        try:
            importances = self.model.feature_importances_
            selected_features = self.feature_selector.get_support()
            
            # Map back to original feature names
            feature_importance = {}
            selected_idx = 0
            
            for i, feature_name in enumerate(self.feature_names):
                if selected_features[i]:
                    feature_importance[feature_name] = importances[selected_idx]
                    selected_idx += 1
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            return dict(sorted_features)
            
        except Exception as e:
            logging.error(f"Error getting feature importance: {e}")
            return {}
    
    def save_model(self):
        """Save the trained model to disk"""
        if not self.is_trained:
            return
        
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_selector': self.feature_selector,
                'feature_names': self.feature_names,
                'training_data_count': len(self.training_data),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            logging.info(f"✅ Model saved to {self.model_file}")
            
        except Exception as e:
            logging.error(f"Error saving model: {e}")
    
    def load_model(self):
        """Load a previously trained model"""
        if not os.path.exists(self.model_file):
            logging.info("No existing model file found")
            return False
        
        try:
            with open(self.model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_selector = model_data['feature_selector']
            self.feature_names = model_data['feature_names']
            
            self.is_trained = True
            
            logging.info(f"✅ Model loaded from {self.model_file}")
            logging.info(f"   Training data count: {model_data.get('training_data_count', 'unknown')}")
            logging.info(f"   Model timestamp: {model_data.get('timestamp', 'unknown')}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return False
    
    def get_model_stats(self) -> Dict:
        """Get model statistics"""
        return {
            'is_trained': self.is_trained,
            'training_samples': len(self.training_data),
            'feature_count': len(self.feature_names),
            'model_file': self.model_file,
            'model_exists': os.path.exists(self.model_file)
        }


class MLInterface:
    """Main ML interface for the tick trading bot"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.feature_engineer = TickFeatureEngineer(
            lookback_ticks=config.get('ml_lookback_ticks', 100)
        )
        self.pattern_recognizer = TradingPatternRecognizer()
        self.ml_model = MLTradingModel(
            model_file=config.get('ml_model_file', 'tick_trading_model.pkl')
        )
        
        # Performance tracking
        self.trade_outcomes = deque(maxlen=1000)
        self.prediction_accuracy = deque(maxlen=100)
        
    def process_tick(self, tick: Dict) -> Dict:
        """Process a new tick and return ML insights"""
        # Add tick to feature engineer
        self.feature_engineer.add_tick(tick)
        
        # Extract features
        features = self.feature_engineer.extract_features()
        
        if not features:
            return {'signal': 'hold', 'confidence': 0.0, 'features': {}}
        
        # Detect patterns
        patterns = self.pattern_recognizer.detect_patterns(features)
        
        # Get ML prediction
        ml_prediction = self.ml_model.predict_trade_signal(features)
        
        # Combine signals
        combined_signal = self._combine_signals(patterns, ml_prediction, features)
        
        return {
            'signal': combined_signal['signal'],
            'confidence': combined_signal['confidence'],
            'ml_prediction': ml_prediction,
            'patterns': patterns,
            'features': features,
            'feature_count': len(features)
        }
    
    def _combine_signals(self, patterns: Dict, ml_prediction: Dict, features: Dict) -> Dict:
        """Combine pattern recognition and ML signals"""
        
        # Start with ML signal if model is trained
        if self.ml_model.is_trained and ml_prediction['confidence'] > 0.6:
            base_signal = ml_prediction['signal']
            base_confidence = ml_prediction['confidence']
        else:
            # Fallback to pattern-based signals
            base_signal = self._get_pattern_signal(patterns)
            base_confidence = max(patterns.values()) if patterns else 0.0
        
        # Apply safety filters
        final_signal = self._apply_safety_filters(base_signal, features, base_confidence)
        
        return final_signal
    
    def _get_pattern_signal(self, patterns: Dict) -> str:
        """Get signal from pattern recognition"""
        if not patterns:
            return 'hold'
        
        # Simple pattern-based logic
        momentum_score = patterns.get('momentum_burst', 0)
        scalping_score = patterns.get('scalping_opportunity', 0)
        breakout_score = patterns.get('breakout_pattern', 0)
        reversal_score = patterns.get('reversal_signal', 0)
        
        # Buy signal conditions
        if (momentum_score > 0.7 and scalping_score > 0.5) or breakout_score > 0.8:
            return 'buy'
        
        # Sell signal conditions  
        if reversal_score > 0.8 or (momentum_score > 0.7 and scalping_score > 0.5):
            return 'sell'
        
        return 'hold'
    
    def _apply_safety_filters(self, signal: str, features: Dict, confidence: float) -> Dict:
        """Apply safety filters to signals"""
        
        # Don't trade if spread is too wide
        if features.get('current_spread', 0) > 0.2:
            return {'signal': 'hold', 'confidence': 0.0}
        
        # Don't trade if volatility is too high
        if features.get('price_volatility', 0) > 0.3:
            return {'signal': 'hold', 'confidence': 0.0}
        
        # Require minimum confidence
        if confidence < 0.5:
            return {'signal': 'hold', 'confidence': confidence}
        
        return {'signal': signal, 'confidence': confidence}
    
    def record_trade_outcome(self, features: Dict, signal: str, profit_loss: float):
        """Record trade outcome for model learning"""
        
        # Determine outcome category
        if signal == 'hold':
            outcome = 'no_trade'
        else:
            outcome = 'profitable' if profit_loss > 0 else 'unprofitable'
        
        # Store outcome
        self.trade_outcomes.append({
            'features': features,
            'signal': signal,
            'outcome': outcome,
            'profit_loss': profit_loss,
            'timestamp': datetime.now()
        })
        
        # Feed to ML model
        self.ml_model.collect_training_data(features, outcome, profit_loss)
        
        # Track prediction accuracy if model made a prediction
        if self.ml_model.is_trained and signal != 'hold':
            predicted_profitable = signal in ['buy', 'sell']  # Simplified
            actual_profitable = profit_loss > 0
            accuracy = 1.0 if predicted_profitable == actual_profitable else 0.0
            self.prediction_accuracy.append(accuracy)
    
    def get_ml_insights(self) -> Dict:
        """Get ML insights and statistics"""
        
        model_stats = self.ml_model.get_model_stats()
        feature_importance = self.ml_model.get_feature_importance()
        
        # Calculate recent accuracy
        recent_accuracy = np.mean(self.prediction_accuracy) if self.prediction_accuracy else 0.0
        
        # Trade outcome statistics
        profitable_trades = sum(1 for outcome in self.trade_outcomes if outcome['profit_loss'] > 0)
        total_trades = len([outcome for outcome in self.trade_outcomes if outcome['signal'] != 'hold'])
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0.0
        
        return {
            'model_trained': model_stats['is_trained'],
            'training_samples': model_stats['training_samples'],
            'recent_accuracy': recent_accuracy,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'feature_importance': dict(list(feature_importance.items())[:10]),  # Top 10 features
            'ml_available': ML_AVAILABLE
        }
    
    def force_retrain(self):
        """Force model retraining"""
        if ML_AVAILABLE:
            logging.info("🤖 Force retraining ML model...")
            success = self.ml_model.train_model(min_samples=20)
            if success:
                logging.info("✅ Model retrained successfully")
            else:
                logging.warning("❌ Model retraining failed")
        else:
            logging.warning("⚠️  ML libraries not available for retraining")


# Configuration for ML Interface
ML_CONFIG = {
    'ml_lookback_ticks': 100,
    'ml_model_file': 'xauusd_tick_ml_model.pkl',
    'auto_retrain_interval': 100,  # Retrain every N trades
    'min_confidence_threshold': 0.6,
    'feature_update_interval': 10  # Update features every N ticks
}