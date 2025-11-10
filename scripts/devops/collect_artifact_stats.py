import json
import os
from pathlib import Path

if __name__ == "__main__":
    if os.environ.get('CI'):
        wildcard = os.environ['ARTIFACT_PATH']
        results = {
            filename: filename.stat().st_size
            for filename in Path('.').glob(wildcard)
        }
        with open(os.environ['STATS_FILE'], 'w') as f:
            json.dump(results, f)
            print(json.dumps(results, indent=2))
