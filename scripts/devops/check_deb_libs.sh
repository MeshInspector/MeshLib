#!/bin/bash

# Checks the libraries a `.deb` installs into one directory
# usage: ./check_deb_libs.sh <path to .deb> <lib dir inside the package>
# e.g.   ./check_deb_libs.sh meshlib-dev.deb /usr/local/lib/MeshLib/lib

set -eo pipefail

stat -c '%n: %s bytes' "$1"
readelf --version > /dev/null

ROOT=$(mktemp -d)
trap 'rm -rf "$ROOT"' EXIT
dpkg --extract "$1" "$ROOT"
LIB_DIR="$ROOT$2"
if [ ! -d "$LIB_DIR" ]; then
  echo "$2 is not in the package"
  exit 1
fi

# absolute or dangling links break on the user's machine
BAD=$(find "$LIB_DIR" -type l \( -lname '/*' -o -xtype l \))
# a regular file under another library's soname makes ldconfig warn "is not a symbolic link"
while read -r f; do
  SONAME=$(readelf -d "$f" 2>/dev/null | sed -n 's/.*Library soname: \[\(.*\)\]/\1/p' || true)
  S="$(dirname "$f")/$SONAME"
  if [ -n "$SONAME" ] && [ "$S" != "$f" ] && [ -f "$S" ] && [ ! -L "$S" ]; then BAD="$BAD"$'\n'"$S"; fi
done < <(find "$LIB_DIR" -type f -name '*.so*')

if [ -n "$BAD" ]; then
  echo "Bad libraries in the package:"
  echo "$BAD" | sed -e '/^$/d' -e "s|^$ROOT||" | sort -u
  exit 1
fi
