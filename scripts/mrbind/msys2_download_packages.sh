#!/bin/bash
set -euo pipefail
cd "$(dirname "$BASH_SOURCE")"
echo 'Downloading packages. This can take a while...'
wget -P msys2_packages -i msys2_package_urls.txt -q --show-progress -c
