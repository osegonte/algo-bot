import json
from pathlib import Path
from sklearn.metrics import accuracy_score  # pseudo — implement later

def score_logs(log_path: Path):
    # TODO: real model; simple profit factor calc for now
    profits = []
    with log_path.open() as f:
        for line in f:
            log = json.loads(line)
            if log["side"] == "sell":
                profits.append(float(log.get("profit", 0)))
    return sum(profits) / len(profits) if profits else 0
