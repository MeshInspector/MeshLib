#!/bin/bash
# Packs the Rust crate `meshlib` (top directory `MeshLibRust`: the committed `Cargo.toml` and `build.rs`, the generated `src`) into the current directory:
#   meshlib_rust_bindings.zip - the crate without libraries; with the NuGet package given, its `native.txt` lets `build.rs` download them;
#   and with the NuGet package only:
#   meshlib_rust_native_<os>-<arch>.zip - the libraries of one platform from the NuGet package, what `build.rs` downloads;
#   meshlib_rust.zip - the crate with the libraries of all platforms in `native/<os>-<arch>`.
# The release renames `meshlib_` to `meshlib_<MeshLib version>_` in all of them.
# Usage: pack_rust_crate.sh <MeshLib version, e.g. v3.1.4.528> <release download URL> <generated src directory> [<MeshLib .nupkg>]
set -euo pipefail

APP_VERSION="$1"
URL="$2"
SRC_DIR="$3"
NUPKG="${4:-}"
ROOT="$(dirname "$(realpath "$0")")/.."
OUT="$PWD"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir "$WORK/MeshLibRust"
cp "$ROOT/source/MeshLibRust/Cargo.toml" "$ROOT/source/MeshLibRust/build.rs" "$ROOT/LICENSE" "$ROOT/thirdparty/licenses/THIRD-PARTY-NOTICES.txt" "$WORK/MeshLibRust/"
cp -r "$SRC_DIR" "$WORK/MeshLibRust/src"

# The crate version is the MeshLib version. Cargo wants semver, so the fourth number becomes build metadata: v3.1.4.528 -> 3.1.4+528.
CRATE_VERSION="$(echo "${APP_VERSION#v}" | sed -E 's/^([0-9]+\.[0-9]+\.[0-9]+)\.([0-9]+)$/\1+\2/')"
sed -i "s/^version = .*/version = \"$CRATE_VERSION\"/" "$WORK/MeshLibRust/Cargo.toml"
grep '^version' "$WORK/MeshLibRust/Cargo.toml"

if [ -n "$NUPKG" ]; then
  unzip -q "$NUPKG" -d "$WORK/nupkg"
  echo "url $URL" > "$WORK/native.txt"
  # NuGet runtime identifier -> `<CARGO_CFG_TARGET_OS>-<CARGO_CFG_TARGET_ARCH>`, see `build.rs`.
  for PAIR in win-x64:windows-x86_64 win-arm64:windows-aarch64 linux-x64:linux-x86_64 linux-arm64:linux-aarch64 osx-x64:macos-x86_64 osx-arm64:macos-aarch64; do
    PLATFORM="${PAIR#*:}"
    LIBS="$WORK/nupkg/runtimes/${PAIR%%:*}/native"
    rm -f "$OUT/meshlib_rust_native_$PLATFORM.zip"
    (cd "$LIBS" && zip -q "$OUT/meshlib_rust_native_$PLATFORM.zip" *)
    echo "$(sha256sum "$OUT/meshlib_rust_native_$PLATFORM.zip" | cut -d' ' -f1) meshlib_${APP_VERSION}_rust_native_$PLATFORM.zip" >> "$WORK/native.txt"
    mkdir -p "$WORK/MeshLibRust/native"
    mv "$LIBS" "$WORK/MeshLibRust/native/$PLATFORM"
  done
  cat "$WORK/native.txt"
  rm -f "$OUT/meshlib_rust.zip"
  (cd "$WORK" && zip -qr "$OUT/meshlib_rust.zip" MeshLibRust)
  rm -r "$WORK/MeshLibRust/native"
  mv "$WORK/native.txt" "$WORK/MeshLibRust/"
fi

rm -f "$OUT/meshlib_rust_bindings.zip"
(cd "$WORK" && zip -qr "$OUT/meshlib_rust_bindings.zip" MeshLibRust)
ls -l "$OUT"/meshlib_rust*.zip
