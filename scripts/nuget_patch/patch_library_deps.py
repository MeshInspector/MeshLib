import sys
from pathlib import Path
import fake_whl_helper as FWH

INPUT_FILE = Path(sys.argv[1])
LIBS_DIR = INPUT_FILE.parent
OUT_DIR = Path(sys.argv[2])

FWH.make_fake_whl(INPUT_FILE)
FWH.patch_whl(OUT_DIR,LIBS_DIR)
