"""
Builds the optional `meshlib-fonts` wheel containing the large CJK font that is
excluded from the main meshlib wheels (it is ~15 MB compressed, 19-24% of a wheel).

The wheel is pure data: it ships `meshlib/NotoSansCJK-Regular.ttc`, which pip
installs into the same site-packages `meshlib/` directory as the main package,
exactly where `SystemPath::getFontsDirectory()` already looks (the wheel's
`__init__.py` overrides the fonts directory to the package directory).
Installable directly or via the `meshlib[fonts]` extra.

The version is independent of the MeshLib version: the font changes very rarely,
and `twine upload --skip-existing` makes re-publishing the same version a no-op.
"""

import base64
import csv
import hashlib
import io
import zipfile

from build_constants import SOURCE_DIR, WHEEL_SCRIPT_DIR

VERSION = "1.0.0"

METADATA = f"""\
Metadata-Version: 2.4
Name: meshlib-fonts
Version: {VERSION}
Summary: Optional CJK font (Noto Sans CJK) for text rendering in MeshLib
License-Expression: OFL-1.1
License-File: THIRD-PARTY-NOTICES.txt
Project-URL: Homepage, https://meshlib.io/
Description-Content-Type: text/markdown

Optional add-on for the [meshlib](https://pypi.org/project/meshlib/) package:
the Noto Sans CJK font used to render Chinese/Japanese/Korean text in the
MeshLib viewer UI and in text-to-mesh functions. Install as
`pip install meshlib[fonts]` or `pip install meshlib-fonts`.
"""

WHEEL_META = """\
Wheel-Version: 1.0
Generator: meshlib-build-fonts-wheel
Root-Is-Purelib: true
Tag: py3-none-any
"""


def record_entry(name, data):
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return [name, f"sha256={digest}", str(len(data))]


def build_fonts_wheel():
    dist_info = f"meshlib_fonts-{VERSION}.dist-info"
    entries = [
        (
            "meshlib/NotoSansCJK-Regular.ttc",
            (SOURCE_DIR / "thirdparty" / "Noto_Sans" / "NotoSansCJK-Regular.ttc").read_bytes(),
        ),
        (
            f"{dist_info}/licenses/THIRD-PARTY-NOTICES.txt",
            (SOURCE_DIR / "thirdparty" / "licenses" / "THIRD-PARTY-NOTICES.txt").read_bytes(),
        ),
        (f"{dist_info}/METADATA", METADATA.encode()),
        (f"{dist_info}/WHEEL", WHEEL_META.encode()),
    ]

    wheel_path = WHEEL_SCRIPT_DIR / f"meshlib_fonts-{VERSION}-py3-none-any.whl"
    with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as wheel:
        record = io.StringIO()
        writer = csv.writer(record, lineterminator="\n")
        for name, data in entries:
            wheel.writestr(name, data)
            writer.writerow(record_entry(name, data))
        writer.writerow([f"{dist_info}/RECORD", "", ""])
        wheel.writestr(f"{dist_info}/RECORD", record.getvalue())

    print(f"Fonts wheel is ready: {wheel_path}")


if __name__ == "__main__":
    build_fonts_wheel()
