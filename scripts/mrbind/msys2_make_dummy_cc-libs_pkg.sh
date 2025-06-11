#!/bin/bash

# MSYS2 has at one point has renamed the `...-gcc-libs` package to `...-cc-libs`: https://github.com/msys2/MINGW-packages/commit/62308009e77d772a126313626b194e503b0e5135
# This has messed up our script that installs the old Clang version.
# This `...-[g]cc-libs` is an alias that `-libc++` provides. So when we install the old Clang, their old `...-libc++` package no longer provides it,
#   which breaks the dependency for the other packages that we DON'T downgrade with the Clang.
# So to work around that, this script builds a dummy `...-cc-libs` package to satisfy this dependency in the packages that we don't downgrade.

# NOTE: This script will become unnecessary when we update Clang to 20 or newer. Then you can destroy it, and also destroy the references to it in `msys2_install_clang_ver.sh`.


set -euxo pipefail

# Generate pkgbuild
echo "pkgname=mingw-w64-clang-x86_64-cc-libs" >PKGBUILD
echo "pkgver=1" >>PKGBUILD
echo "pkgrel=1" >>PKGBUILD
echo "pkgdesc="Dummy placeholder"" >>PKGBUILD
echo "arch=('any')" >>PKGBUILD
echo "url=""" >>PKGBUILD
echo "license=\"custom\"" >>PKGBUILD
echo "depends=()" >>PKGBUILD
echo "source=()" >>PKGBUILD
echo "build(){" >>PKGBUILD
echo "    true" >>PKGBUILD
echo "}" >>PKGBUILD

# Disable stripping in `/etc/makepkg.conf`, to avoid having to install `strip`.
# We do this temporarily and then restore the old config.

# First, roll apply the existing backup if any, to avoid permantly messing up the config.
# If the backup exists, it means the process didn't complete properly the last time.
[[ -f /etc/makepkg.conf.before_dummy_pkg_build ]] && mv /etc/makepkg.conf.before_dummy_pkg_build /etc/makepkg.conf

# Now patch `makepkg.conf`, and back up the old one.
awk '/^OPTIONS=/{$0=gensub(/([^!])strip/,"\\1!strip",1)} {print}' /etc/makepkg.conf >/etc/makepkg.conf.new
mv /etc/makepkg.conf /etc/makepkg.conf.before_dummy_pkg_build
mv /etc/makepkg.conf.new /etc/makepkg.conf

# Build the package!
makepkg

# Lastly, roll back the backup
mv /etc/makepkg.conf.before_dummy_pkg_build /etc/makepkg.conf