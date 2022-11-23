#!/bin/bash
# add thirdparty libs to ld.so.conf

# get absolute path
SCRIPT_DIR=$(dirname "$0")
THIRDPARTY_LIB_DIR=$(realpath "$SCRIPT_DIR/../lib")

# do not add the directory if it already exists
grep -qxF "${THIRDPARTY_LIB_DIR}" /etc/ld.so.conf || \
  echo "${THIRDPARTY_LIB_DIR}" | sudo tee -a  /etc/ld.so.conf && sudo ldconfig

exit 0
