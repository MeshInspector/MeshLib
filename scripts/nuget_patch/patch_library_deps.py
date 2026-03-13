import sys
from pathlib import Path
import fake_whl_helper as FWH

LIBS_DIR = Path(sys.argv[1])
INPUT_FILES = (LIBS_DIR / Path(x) for x in sys.argv[2].split(":"))
OUT_DIR = Path(sys.argv[3])

FWH.make_fake_whl(INPUT_FILES)
FWH.patch_whl(OUT_DIR,LIBS_DIR)
