import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def score_logs(log_path: Path):
    """Original simple scoring function"""
    profits = []
    with log_path.open() as f:
        for line in f:
            log = json.loads(line)
            if log["side"] == "sell":
                profits.append(float(log.get("profit", 0)))
    return sum(profits) / len(profits) if profits else 0

def score_strategies():
    """Level 1: Rule-based strategy scoring and ranking"""
    
    # Define available strategies
    strategies = ["martingale", "breakout", "mean_reversion"]
    
    # Simulate performance data for each strategy
    # In production, this would analyze actual trade logs per strategy
    strategy_data = []
    
    for strategy in strategies:
        # Simulate strategy performance metrics
        np.random.seed(hash(strategy) % 1000)  # Consistent seed per strategy
        
        # Generate simulated trading results
        num_trades = np.random.randint(20, 50)
        
        if strategy == "breakout":
            # Breakout tends to have higher profit factor but lower win rate
            win_rate = np.random.uniform(0.35, 0.45)
            avg_win = np.random.uniform(25, 40)
            avg_loss = np.random.uniform(-15, -10)
            sharpe = np.random.uniform(0.8, 1.2)
            
        elif strategy == "mean_reversion":
            # Mean reversion tends to have higher win rate but smaller wins
            win_rate = np.random.uniform(0.55, 0.70)
            avg_win = np.random.uniform(8, 15)
            avg_loss = np.random.uniform(-12, -8)
            sharpe = np.random.uniform(0.6, 1.0)
            
        elif strategy == "martingale":
            # Martingale has high win rate but catastrophic losses
            win_rate = np.random.uniform(0.75, 0.85)
            avg_win = np.random.uniform(5, 10)
            avg_loss = np.random.uniform(-50, -30)
            sharpe = np.random.uniform(0.2, 0.6)
        
        # Calculate derived metrics
        gross_profit = num_trades * win_rate * avg_win
        gross_loss = num_trades * (1 - win_rate) * avg_loss
        net_pnl = gross_profit + gross_loss  # avg_loss is negative
        profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else float('inf')
        
        # Rule-based scoring system
        score = calculate_strategy_score(
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            net_pnl=net_pnl,
            num_trades=num_trades
        )
        
        strategy_data.append({
            "strategy": strategy,
            "win_rate": round(win_rate * 100, 1),
            "profit_factor": round(profit_factor, 2),
            "sharpe_ratio": round(sharpe, 3),
            "net_pnl": round(net_pnl, 2),
            "num_trades": num_trades,
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "score": round(score, 2),
            "last_updated": datetime.utcnow().isoformat()
        })
    
    # Create DataFrame and sort by score
    df = pd.DataFrame(strategy_data)
    df = df.sort_values('score', ascending=False)
    
    # Display results
    print("\n" + "="*70)
    print("🤖 STRATEGY SCORING & RANKING (Level 1)")
    print("="*70)
    print(df[['strategy', 'score', 'win_rate', 'profit_factor', 'sharpe_ratio', 'net_pnl']].to_string(index=False))
    
    print(f"\n🏆 Top 3 Strategies:")
    for i, (_, row) in enumerate(df.head(3).iterrows(), 1):
        print(f"  {i}. {row['strategy'].title()}: Score {row['score']} "
              f"(Win Rate: {row['win_rate']}%, PF: {row['profit_factor']})")
    
    # Save results
    save_strategy_scores(df)
    
    return df

def calculate_strategy_score(win_rate, profit_factor, sharpe_ratio, net_pnl, num_trades):
    """Rule-based scoring algorithm"""
    
    # Weights for different metrics (can be tuned)
    weights = {
        'win_rate': 0.25,
        'profit_factor': 0.30,
        'sharpe_ratio': 0.25,
        'consistency': 0.20
    }
    
    # Normalize metrics to 0-100 scale
    
    # Win rate (already 0-1, convert to 0-100)
    win_score = min(win_rate * 100, 100)
    
    # Profit factor (normalize using sigmoid-like function)
    pf_score = min((profit_factor - 1) * 50, 100) if profit_factor >= 1 else 0
    
    # Sharpe ratio (normalize, good sharpe is >1.0)
    sharpe_score = min(max(sharpe_ratio * 50, 0), 100)
    
    # Consistency score (based on number of trades and avoiding extreme strategies)
    consistency_score = min(num_trades * 2, 100)  # More trades = more consistent
    
    # Penalize strategies with extreme characteristics
    if profit_factor > 5:  # Too good to be true
        consistency_score *= 0.8
    if win_rate > 0.9:  # Suspiciously high win rate
        consistency_score *= 0.7
    
    # Calculate weighted score
    final_score = (
        weights['win_rate'] * win_score +
        weights['profit_factor'] * pf_score +
        weights['sharpe_ratio'] * sharpe_score +
        weights['consistency'] * consistency_score
    )
    
    return final_score

def save_strategy_scores(df):
    """Save strategy scores to file for later use"""
    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)
    
    # Save as JSON for easy consumption by other modules
    scores_file = log_path / "strategy_scores.json"
    
    scores_data = {
        "generated_at": datetime.utcnow().isoformat(),
        "scores": df.to_dict('records')
    }
    
    with open(scores_file, 'w') as f:
        json.dump(scores_data, f, indent=2)
    
    print(f"💾 Strategy scores saved to: {scores_file}")

def get_top_strategy():
    """Get the highest scoring strategy"""
    try:
        with open("logs/strategy_scores.json") as f:
            data = json.load(f)
        
        if data['scores']:
            return data['scores'][0]['strategy']  # First item is highest scored
    except FileNotFoundError:
        pass
    
    return "breakout"  # Default fallback

if __name__ == "__main__":
    # Run Level 1 scoring
    df = score_strategies()
    print(f"\n🎯 Recommended strategy: {get_top_strategy()}")