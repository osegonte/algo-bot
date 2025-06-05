from pathlib import Path
import shutil

CHILD_LOG_DIR = Path("logs")
PARENT_DIR = Path("../parent_bot/logs")  # adjust if running both locally

def push_logs():
    PARENT_DIR.mkdir(parents=True, exist_ok=True)
    for file in CHILD_LOG_DIR.glob("*.json"):
        shutil.copy(file, PARENT_DIR / file.name)

if __name__ == "__main__":
    push_logs()
