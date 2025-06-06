#!/usr/bin/env python3
"""
Level 7-D: Hybrid Ranking System
Combines rule-based and ML scores: final_score = 0.5*rule_score + 0.5*ml_score
"""

import pandas as pd
import numpy as np
import json
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class HybridRankingSystem:
    """Hybrid ranking system combining rule-based and ML predictions"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Directories
        self.logs_dir = Path("logs")
        self.config_dir = Path("config")
        
        # Hybrid scoring configuration
        self.hybrid_config = self.config.get('hybrid_ranking', {
            'rule_weight': 0.5,
            'ml_weight': 0.5,
            'min_ml_confidence': 0.3,
            'fallback_to_rules': True,
            'score_normalization': True
        })
        
        # Load weights from config file if available
        self._load_hybrid_weights()
        
    def _load_hybrid_weights(self):
        """Load hybrid weights from YAML config"""
        
        config_file = self.config_dir / "hybrid_config.yaml"
        
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config_data = yaml.safe_load(f)
                
                hybrid_settings = config_data.get('hybrid_ranking', {})
                
                # Update weights if specified
                if 'rule_weight' in hybrid_settings:
                    self.hybrid_config['rule_weight'] = hybrid_settings['rule_weight']
                if 'ml_weight' in hybrid_settings:
                    self.hybrid_config['ml_weight'] = hybrid_settings['ml_weight']
                
                print(f"📊 Loaded hybrid weights: rule={self.hybrid_config['rule_weight']}, ml={self.hybrid_config['ml_weight']}")
                
            except Exception as e:
                print(f"⚠️ Error loading hybrid config: {e}")
        else:
            # Create default config file
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default hybrid configuration file"""
        
        default_config = {
            'hybrid_ranking': {
                'rule_weight': 0.5,
                'ml_weight': 0.5,
                'min_ml_confidence': 0.3,
                'fallback_to_rules': True,
                'score_normalization': True,
                'description': 'Hybrid ranking configuration - adjust weights as needed'
            }
        }
        
        self.config_dir.mkdir(exist_ok=True)
        config_file = self.config_dir / "hybrid_config.yaml"
        
        with open(config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        print(f"📝 Created default hybrid config: {config_file}")
    
    def load_rule_based_scores(self) -> Optional[pd.DataFrame]:
        """Load latest rule-based strategy scores"""
        
        scores_file = self.logs_dir / "strategy_scores.json"
        
        if not scores_file.exists():
            print("⚠️ No rule-based scores found")
            return None
        
        try:
            with open(scores_file) as f:
                scores_data = json.load(f)
            
            scores = scores_data.get('scores', [])
            
            if not scores:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(scores)
            
            # Ensure required columns
            required_cols = ['strategy', 'score', 'win_rate', 'profit_factor', 'net_pnl']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"⚠️ Missing rule score columns: {missing_cols}")
                return None
            
            print(f"✅ Loaded {len(df)} rule-based scores")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading rule scores: {e}")
            return None
    
    def load_ml_predictions(self) -> Optional[pd.DataFrame]:
        """Load latest ML predictions"""
        
        ml_file = self.logs_dir / "latest_ml_predictions.json"
        
        if not ml_file.exists():
            print("⚠️ No ML predictions found")
            return None
        
        try:
            with open(ml_file) as f:
                ml_data = json.load(f)
            
            predictions = ml_data.get('predictions', [])
            
            if not predictions:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(predictions)
            
            # Ensure required columns
            required_cols = ['strategy', 'ml_score', 'ml_confidence']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"⚠️ Missing ML prediction columns: {missing_cols}")
                return None
            
            print(f"✅ Loaded {len(df)} ML predictions")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading ML predictions: {e}")
            return None
    
    def generate_hybrid_rankings(self) -> pd.DataFrame:
        """Generate hybrid rankings combining rule and ML scores"""
        
        print("🔀 Generating Hybrid Rankings...")
        print("=" * 40)
        
        # Load both scoring systems
        rule_scores = self.load_rule_based_scores()
        ml_predictions = self.load_ml_predictions()
        
        # Determine available strategies
        strategies = ['martingale', 'breakout', 'mean_reversion']
        
        if rule_scores is not None:
            strategies = rule_scores['strategy'].tolist()
        elif ml_predictions is not None:
            strategies = ml_predictions['strategy'].tolist()
        
        hybrid_results = []
        
        for strategy in strategies:
            
            # Get rule-based score
            rule_score = 50.0  # Default neutral score
            rule_data = {}
            
            if rule_scores is not None:
                rule_row = rule_scores[rule_scores['strategy'] == strategy]
                if not rule_row.empty:
                    rule_score = float(rule_row.iloc[0]['score'])
                    rule_data = {
                        'rule_win_rate': float(rule_row.iloc[0]['win_rate']),
                        'rule_profit_factor': float(rule_row.iloc[0]['profit_factor']),
                        'rule_net_pnl': float(rule_row.iloc[0]['net_pnl']),
                        'rule_trades': int(rule_row.iloc[0].get('num_trades', 0))
                    }
            
            # Get ML score
            ml_score = 50.0  # Default neutral score
            ml_confidence = 0.1  # Low confidence for fallback
            ml_data = {}
            
            if ml_predictions is not None:
                ml_row = ml_predictions[ml_predictions['strategy'] == strategy]
                if not ml_row.empty:
                    ml_score = float(ml_row.iloc[0]['ml_score'])
                    ml_confidence = float(ml_row.iloc[0]['ml_confidence'])
                    ml_data = {
                        'ml_prediction': float(ml_row.iloc[0].get('ml_prediction', 0)),
                        'ml_confidence': ml_confidence
                    }
            
            # Calculate hybrid score
            hybrid_result = self._calculate_hybrid_score(
                strategy, rule_score, ml_score, ml_confidence, rule_data, ml_data
            )
            
            hybrid_results.append(hybrid_result)
        
        # Convert to DataFrame and sort
        df = pd.DataFrame(hybrid_results)
        df = df.sort_values('hybrid_score', ascending=False).reset_index(drop=True)
        
        # Display results
        self._display_hybrid_rankings(df)
        
        # Save results
        self._save_hybrid_rankings(df)
        
        return df
    
    def _calculate_hybrid_score(self, strategy: str, rule_score: float, ml_score: float, 
                              ml_confidence: float, rule_data: Dict, ml_data: Dict) -> Dict:
        """Calculate hybrid score for a strategy"""
        
        # Normalize scores to 0-100 range if needed
        if self.hybrid_config['score_normalization']:
            rule_score = max(0, min(100, rule_score))
            ml_score = max(0, min(100, ml_score))
        
        # Check ML confidence threshold
        use_ml = ml_confidence >= self.hybrid_config['min_ml_confidence']
        
        if use_ml:
            # Standard hybrid calculation
            hybrid_score = (
                self.hybrid_config['rule_weight'] * rule_score +
                self.hybrid_config['ml_weight'] * ml_score
            )
            scoring_method = 'hybrid'
        else:
            # Fallback to rules if ML confidence too low
            if self.hybrid_config['fallback_to_rules']:
                hybrid_score = rule_score
                scoring_method = 'rule_fallback'
            else:
                # Use hybrid with low-confidence ML
                hybrid_score = (
                    self.hybrid_config['rule_weight'] * rule_score +
                    self.hybrid_config['ml_weight'] * ml_score
                )
                scoring_method = 'hybrid_low_conf'
        
        # Compile result
        result = {
            'strategy': strategy,
            'rule_score': rule_score,
            'ml_score': ml_score,
            'ml_confidence': ml_confidence,
            'hybrid_score': round(hybrid_score, 2),
            'scoring_method': scoring_method,
            'rule_weight_used': self.hybrid_config['rule_weight'],
            'ml_weight_used': self.hybrid_config['ml_weight'],
            'generated_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Add detailed data
        result.update(rule_data)
        result.update(ml_data)
        
        return result
    
    def _display_hybrid_rankings(self, df: pd.DataFrame):
        """Display hybrid rankings with both subscores"""
        
        print(f"\n🏆 HYBRID STRATEGY RANKINGS")
        print("=" * 50)
        print(f"Weights: Rule={self.hybrid_config['rule_weight']:.1f}, ML={self.hybrid_config['ml_weight']:.1f}")
        print("-" * 50)
        
        for i, row in df.iterrows():
            # Confidence indicator
            conf = row['ml_confidence']
            conf_indicator = "⭐" if conf > 0.8 else "🔸" if conf > 0.6 else "🔹" if conf > 0.3 else "❓"
            
            # Scoring method indicator
            method = row['scoring_method']
            method_indicator = "🔀" if method == 'hybrid' else "📏" if method == 'rule_fallback' else "⚠️"
            
            print(f"{i+1}. {row['strategy'].upper()}: {row['hybrid_score']:.1f} {method_indicator}")
            print(f"   Rule: {row['rule_score']:.1f} | ML: {row['ml_score']:.1f} {conf_indicator}")
            
            # Additional details if available
            if 'rule_win_rate' in row:
                print(f"   Win Rate: {row['rule_win_rate']:.1f}% | P&L: ${row.get('rule_net_pnl', 0):.1f}")
            
            print()
        
        # Show top recommendation
        top_strategy = df.iloc[0]
        print(f"🎯 TOP RECOMMENDATION: {top_strategy['strategy'].upper()}")
        print(f"   Hybrid Score: {top_strategy['hybrid_score']:.1f}")
        print(f"   Method: {top_strategy['scoring_method'].replace('_', ' ').title()}")
        
        if top_strategy['ml_confidence'] > 0.7:
            print(f"   High ML Confidence: {top_strategy['ml_confidence']:.2f} ⭐")
        elif top_strategy['ml_confidence'] < 0.4:
            print(f"   Low ML Confidence: {top_strategy['ml_confidence']:.2f} - Rule-driven")
    
    def _save_hybrid_rankings(self, df: pd.DataFrame):
        """Save hybrid rankings to file"""
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.logs_dir / f"hybrid_rankings_{timestamp}.json"
        
        rankings_data = {
            'rankings': df.to_dict('records'),
            'config': self.hybrid_config,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'top_strategy': df.iloc[0]['strategy'] if len(df) > 0 else None
        }
        
        with open(results_file, 'w') as f:
            json.dump(rankings_data, f, indent=2)
        
        # Save as latest
        latest_file = self.logs_dir / "latest_hybrid_rankings.json"
        with open(latest_file, 'w') as f:
            json.dump(rankings_data, f, indent=2)
        
        # Save simplified version for parent controller
        parent_rankings = []
        for _, row in df.iterrows():
            parent_rankings.append({
                'strategy': row['strategy'],
                'hybrid_score': row['hybrid_score'],
                'rule_score': row['rule_score'],
                'ml_score': row['ml_score'],
                'ml_confidence': row['ml_confidence'],
                'rank': len(parent_rankings) + 1
            })
        
        parent_file = self.logs_dir / "strategy_rankings_hybrid.json"
        with open(parent_file, 'w') as f:
            json.dump({
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'rankings': parent_rankings,
                'weights': {
                    'rule_weight': self.hybrid_config['rule_weight'],
                    'ml_weight': self.hybrid_config['ml_weight']
                }
            }, f, indent=2)
        
        print(f"💾 Rankings saved:")
        print(f"   Detailed: {results_file.name}")
        print(f"   Latest: {latest_file.name}")
        print(f"   Parent: {parent_file.name}")
    
    def update_hybrid_weights(self, rule_weight: float, ml_weight: float) -> bool:
        """Update hybrid scoring weights"""
        
        # Validate weights
        if rule_weight < 0 or ml_weight < 0:
            print("❌ Weights must be non-negative")
            return False
        
        total_weight = rule_weight + ml_weight
        if total_weight == 0:
            print("❌ At least one weight must be positive")
            return False
        
        # Normalize weights to sum to 1
        rule_weight = rule_weight / total_weight
        ml_weight = ml_weight / total_weight
        
        # Update configuration
        self.hybrid_config['rule_weight'] = rule_weight
        self.hybrid_config['ml_weight'] = ml_weight
        
        # Save to config file
        config_file = self.config_dir / "hybrid_config.yaml"
        config_data = {
            'hybrid_ranking': {
                'rule_weight': rule_weight,
                'ml_weight': ml_weight,
                'min_ml_confidence': self.hybrid_config['min_ml_confidence'],
                'fallback_to_rules': self.hybrid_config['fallback_to_rules'],
                'score_normalization': self.hybrid_config['score_normalization'],
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        
        print(f"✅ Updated hybrid weights: rule={rule_weight:.2f}, ml={ml_weight:.2f}")
        
        return True
    
    def analyze_score_distribution(self, df: pd.DataFrame) -> Dict:
        """Analyze the distribution of rule vs ML scores"""
        
        analysis = {
            'score_correlation': float(df['rule_score'].corr(df['ml_score'])),
            'rule_score_stats': {
                'mean': float(df['rule_score'].mean()),
                'std': float(df['rule_score'].std()),
                'min': float(df['rule_score'].min()),
                'max': float(df['rule_score'].max())
            },
            'ml_score_stats': {
                'mean': float(df['ml_score'].mean()),
                'std': float(df['ml_score'].std()),
                'min': float(df['ml_score'].min()),
                'max': float(df['ml_score'].max())
            },
            'confidence_stats': {
                'mean': float(df['ml_confidence'].mean()),
                'min': float(df['ml_confidence'].min()),
                'high_conf_count': int((df['ml_confidence'] > 0.7).sum()),
                'low_conf_count': int((df['ml_confidence'] < 0.4).sum())
            }
        }
        
        return analysis

def main():
    """Command line interface for hybrid ranking"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid Ranking System (Level 7-D)")
    parser.add_argument('--generate', action='store_true', help='Generate hybrid rankings')
    parser.add_argument('--weights', nargs=2, type=float, metavar=('RULE', 'ML'), 
                       help='Update hybrid weights (e.g., --weights 0.6 0.4)')
    parser.add_argument('--analyze', action='store_true', help='Analyze score distributions')
    parser.add_argument('--config', action='store_true', help='Show current configuration')
    
    args = parser.parse_args()
    
    print("🔀 Hybrid Ranking System (Level 7-D)")
    print("=" * 50)
    
    ranker = HybridRankingSystem()
    
    if args.config:
        # Show current configuration
        print(f"📊 Current Configuration:")
        print(f"   Rule Weight: {ranker.hybrid_config['rule_weight']:.2f}")
        print(f"   ML Weight: {ranker.hybrid_config['ml_weight']:.2f}")
        print(f"   Min ML Confidence: {ranker.hybrid_config['min_ml_confidence']:.2f}")
        print(f"   Fallback to Rules: {ranker.hybrid_config['fallback_to_rules']}")
        return
    
    if args.weights:
        # Update weights
        rule_weight, ml_weight = args.weights
        success = ranker.update_hybrid_weights(rule_weight, ml_weight)
        if success:
            print("✅ Weights updated successfully")
        return
    
    if args.generate or not any([args.weights, args.analyze, args.config]):
        # Generate hybrid rankings
        df = ranker.generate_hybrid_rankings()
        
        if args.analyze and len(df) > 0:
            # Analyze score distributions
            analysis = ranker.analyze_score_distribution(df)
            
            print(f"\n📊 Score Distribution Analysis:")
            print(f"   Rule-ML Correlation: {analysis['score_correlation']:.3f}")
            print(f"   Rule Scores: {analysis['rule_score_stats']['mean']:.1f} ± {analysis['rule_score_stats']['std']:.1f}")
            print(f"   ML Scores: {analysis['ml_score_stats']['mean']:.1f} ± {analysis['ml_score_stats']['std']:.1f}")
            print(f"   High Confidence Predictions: {analysis['confidence_stats']['high_conf_count']}/{len(df)}")
        
        print(f"\n✅ Level 7-D Complete!")
        print(f"🔀 Hybrid rankings generated with configurable weights")
        print(f"🎯 Ready for Level 7-E: Model Monitoring")

if __name__ == "__main__":
    main()