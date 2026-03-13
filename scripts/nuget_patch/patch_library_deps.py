import sys
from pathlib import Path
import fake_whl_helper as FWH

OUT_DIR = Path(sys.argv[1])
INPUT_FILES = [Path(x) for x in sys.argv[2:]]

FWH.make_fake_whl(INPUT_FILES)
FWH.patch_whl(OUT_DIR,list(set(x.parent for x in INPUT_FILES)))
