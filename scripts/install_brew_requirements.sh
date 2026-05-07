#!/bin/bash

# This script installs requirements by `brew` if not already installed

BASEDIR=$(dirname $(realpath "$0"))
MESHLIB_BREW_REQUIREMENTS=$(cat "$BASEDIR"/../requirements/macos.txt)
if [ -n "$MESHLIB_EXTRA_BREW_REQUIREMENTS" ] ; then
  MESHLIB_BREW_REQUIREMENTS=$MESHLIB_BREW_REQUIREMENTS$'\n'$MESHLIB_EXTRA_BREW_REQUIREMENTS
fi

brew install --quiet $(echo "$MESHLIB_BREW_REQUIREMENTS" | tr '\n' ' ')

brew install --quiet pybind11

# === DIAGNOSTIC: brew strip investigation ===
# A previous attempt (#6063 v1) found that strip+codesign on brew Cellar
# yielded byte-identical bundled dylibs in the wheel — the strip appeared
# to be a silent no-op. Instrument here to confirm whether bytes change
# on Cellar, and which exact files brew has installed.
BREW_PREFIX=$(brew --prefix)
SAMPLE_GLOB=("$BREW_PREFIX"/Cellar/openvdb/*/lib/libopenvdb.*.dylib)
echo "::group::strip-diagnostic: pre-strip"
echo "BREW_PREFIX=$BREW_PREFIX"
echo "Cellar dylibs total count: $(find "$BREW_PREFIX/Cellar" -type f -name '*.dylib' | wc -l)"
echo "Cellar dylibs total size: $(find "$BREW_PREFIX/Cellar" -type f -name '*.dylib' -exec du -b {} + | awk '{s+=$1} END {print s}') bytes"
echo "Sample (libopenvdb*) before strip:"
ls -la "${SAMPLE_GLOB[@]}" 2>&1 || true
shasum -a 256 "${SAMPLE_GLOB[@]}" 2>&1 || true
echo "  codesign verify on sample:"
for f in "${SAMPLE_GLOB[@]}"; do
  [ -f "$f" ] || continue
  codesign --display --verbose=2 "$f" 2>&1 | head -5 || true
done
echo "::endgroup::"

echo "::group::strip-diagnostic: running strip + codesign"
# Verbose: each `-exec` separately (not combined) so we see strip's stderr per file
find "$BREW_PREFIX/Cellar" -type f -name '*.dylib' -print0 | while IFS= read -r -d '' f; do
  chmod u+w "$f"
  strip -x "$f" 2>&1 | grep -v '^$' | sed "s|^|  strip: |" || true
  codesign --force --sign - "$f" 2>&1 | grep -v '^$' | sed "s|^|  codesign: |" || true
done
echo "::endgroup::"

echo "::group::strip-diagnostic: post-strip"
echo "Cellar dylibs total size: $(find "$BREW_PREFIX/Cellar" -type f -name '*.dylib' -exec du -b {} + | awk '{s+=$1} END {print s}') bytes"
echo "Sample (libopenvdb*) after strip:"
ls -la "${SAMPLE_GLOB[@]}" 2>&1 || true
shasum -a 256 "${SAMPLE_GLOB[@]}" 2>&1 || true
echo "  codesign verify on sample:"
for f in "${SAMPLE_GLOB[@]}"; do
  [ -f "$f" ] || continue
  codesign --display --verbose=2 "$f" 2>&1 | head -5 || true
done
echo "::endgroup::"

# check and upgrade python3 pip
python3.10 -m ensurepip --upgrade
python3.10 -m pip install --upgrade pip

# install requirements for python libs
python3.10 -m pip install -r requirements/python.txt

exit 0
