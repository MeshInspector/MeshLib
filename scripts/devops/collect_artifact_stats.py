import json
import os
from pathlib import Path

if __name__ == "__main__":
    if os.environ.get('CI'):
        path = Path(os.environ.get('ARTIFACT_PATH', '.'))
        glob = os.environ['ARTIFACT_GLOB']
        results = {
            str(filename.relative_to(path)): filename.stat().st_size
            for filename in path.glob(glob)
        }
        with open(os.environ['STATS_FILE'], 'w') as f:
            json.dump(results, f)
            print(json.dumps(results, indent=2))
