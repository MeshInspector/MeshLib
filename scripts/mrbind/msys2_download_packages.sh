#!/bin/bash
set -euo pipefail
cd "$(dirname "$BASH_SOURCE")"
wget -P msys2_packages -i msys2_package_urls.txt -q --show-progress -c
