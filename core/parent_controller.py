import json
import glob
from pathlib import Path

LOG_PATH = Path("logs")

class ParentController:
    def __init__(self):
        self.logs = []

    def ingest_logs(self):
        for logfile in LOG_PATH.glob("*.json"):
            with open(logfile) as f:
                for line in f:
                    self.logs.append(json.loads(line))

    def basic_stats(self):
        # very naive P&L estimator
        buys = [l for l in self.logs if l["side"] == "buy"]
        sells = [l for l in self.logs if l["side"] == "sell"]
        print(f"Total buys: {len(buys)} | sells: {len(sells)}")

if __name__ == "__main__":
    pc = ParentController()
    pc.ingest_logs()
    pc.basic_stats()
