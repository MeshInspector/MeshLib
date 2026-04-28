import platform
import subprocess
import sys
from pathlib import Path
import fake_whl_helper as FWH

OUT_DIR = Path(sys.argv[1])
INPUT_FILES = [Path(x) for x in sys.argv[2:]]

if platform.system() == "Linux":
    # Drop libz-ng.so.2 ZLIB_NG_* version requirements from every .so
    # auditwheel will see -- the input C-API libs and any dep .so in their
    # parent dirs that auditwheel will pull in during repair. auditwheel has
    # no manylinux policy entry for libz-ng, so leaving the tags in place
    # fails the policy check.
    # See scripts/wheel/strip_zlib_ng_symbol_versions.py for details.
    SCRIPT_ROOT = Path(__file__).resolve().parent.parent
    subprocess.check_call([
        sys.executable,
        str(SCRIPT_ROOT / "wheel" / "strip_zlib_ng_symbol_versions.py"),
        *sorted({str(x.parent) for x in INPUT_FILES}),
    ])

FWH.make_fake_whl(INPUT_FILES)
FWH.patch_whl(OUT_DIR,list(set(x.parent for x in INPUT_FILES)))
