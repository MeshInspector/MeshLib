#!/bin/bash

# This script installs a specific Clang version in MSYS2.
# You shouldn't need to call this directly, this is called by `install_deps_windows_msys2.bat`.
#
# The existing Clang version, if any, is uninstalled.
# The packages are downloaded to `~/clang_downloads`. If they already exist, they aren't downloaded again.

set -euxo pipefail

# Check that the number of arguments (`$#`) is one, and that the first argument (`$1`) isn't empty.
[[ $# == 1 && $1 != "" ]] || (echo "Must specify one argument, the desired Clang version."; exit 1)

# Abort if already installed.
if [[ $(LANG= pacman -Qi $MINGW_PACKAGE_PREFIX-clang 2>/dev/null | grep '^Version ' | awk '{print $3}') == $1 ]]; then
    echo "Clang $1 is already installed, nothing to do."
    exit 0
fi

DOWNLOAD_DIR="$(realpath ~/clang_downloads)"

# This list should match what you'd see by installing `$MINGW_PACKAGE_PREFIX-{clang,clang-tools-extra}` and then looking at the list of packages (in `pacman -Q`).
# But I could've included some unnecessary parts.
PACKAGES="$(echo $MINGW_PACKAGE_PREFIX-{clang,clang-libs,clang-tools-extra,compiler-rt,libc++,libunwind,lld,llvm,llvm-libs})"

mkdir -p "$DOWNLOAD_DIR"

# Download packages.
PACKAGE_FILES=
for x in $PACKAGES; do
    wget -q --show-progress -c "https://mirror.msys2.org/mingw/clang64/$x-$1-any.pkg.tar.zst" -O "$DOWNLOAD_DIR/$x-$1-any.pkg.tar.zst"
    PACKAGE_FILES+=" $DOWNLOAD_DIR/$x-$1-any.pkg.tar.zst"
done


# Generate the stub for the `cc-libs` package if needed. See `msys2_make_dummy_cc-libs_pkg.sh` for details.
# [
if [[ $1 = 18.* || $1 = 19.* ]]; then
    SCRIPT_DIR="$(realpath "$(dirname "$BASH_SOURCE")")"
    "$SCRIPT_DIR/msys2_make_dummy_cc-libs_pkg.sh" "$DOWNLOAD_DIR"
    PACKAGE_FILES+=" $DOWNLOAD_DIR/mingw-w64-clang-x86_64-cc-libs-1-1-any.pkg.tar.zst"
else
    # If you see this message, then this specific workaround is no longer necessary. Please destroy this entire `[...]` code block,
    #   and destroy the `msys2_make_dummy_cc-libs_pkg.sh` script.
    echo "### NOTE: Now when we've updated Clang to 20+, the cc-libs stub workaround is no longer necessary! Please remove it from the `msys2_install_clang_ver.sh` script."
fi
# ]


# Install packages. This will automatically replace existing packages, if any.
pacman -U --noconfirm $PACKAGE_FILES

# Configure Pacman no never update Clang.
sed -i 's/^\(\|#\)IgnorePkg\b.*/IgnorePkg = '"$PACKAGES"'/' /etc/pacman.conf
