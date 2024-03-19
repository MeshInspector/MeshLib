#!/bin/bash
set -eo pipefail

SCRIPT_DIR=$(dirname $(realpath "$0"))
INSTALL_DIR="$1"

LIBRARIES=(
  occt
)

for LIBRARY in $LIBRARIES ; do
  "$SCRIPT_DIR/$LIBRARY.sh" "$INSTALL_DIR"
done
