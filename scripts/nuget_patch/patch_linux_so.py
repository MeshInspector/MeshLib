import sys
from pathlib import Path
import fake_whl_helper as FWH

SO_DIR = Path(sys.argv[1])
OUT_DIR = Path(sys.argv[2])

FWH.make_fake_whl(SO_DIR / "libMRMeshC.so")
FWH.patch_whl(OUT_DIR,SO_DIR)
