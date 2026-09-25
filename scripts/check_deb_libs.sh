#!/bin/bash

# Checks the thirdparty libs of a `.deb` built by `distribution.sh` and extracted with `dpkg --extract`
# usage: ./check_deb_libs.sh <path to .deb> <extraction root>

set -eo pipefail

stat -c '%n: %s bytes' "$1"
LIB_DIR="$2/usr/local/lib/MeshLib/lib"
readelf --version > /dev/null

# absolute or dangling links break on the user's machine
BAD=$(find "$LIB_DIR" -type l \( -lname '/*' -o -xtype l \))
# a regular file under another library's soname makes ldconfig warn "is not a symbolic link"
while read -r f; do
  SONAME=$(readelf -d "$f" 2>/dev/null | sed -n 's/.*Library soname: \[\(.*\)\]/\1/p' || true)
  S="$(dirname "$f")/$SONAME"
  if [ -n "$SONAME" ] && [ "$S" != "$f" ] && [ -f "$S" ] && [ ! -L "$S" ]; then BAD="$BAD"$'\n'"$S"; fi
done < <(find "$LIB_DIR" -type f -name '*.so*')

if [ -n "$BAD" ]; then
  echo "Bad libraries in the package:$BAD"
  exit 1
fi
