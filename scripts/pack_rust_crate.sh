#!/bin/bash
# Packs the Rust crate `meshlib` into a zip with the top directory `MeshLibRust`: the committed `Cargo.toml` and `build.rs`,
# the generated `src`, and, if the NuGet package is given, its native libraries of all platforms in `native/<os>-<arch>`.
# Usage: pack_rust_crate.sh <MeshLib version, e.g. v3.1.4.528> <generated src directory> <output zip> [<MeshLib .nupkg>]
set -euo pipefail

VERSION="${1#v}"
SRC_DIR="$2"
OUTPUT="$(realpath -m "$3")"
NUPKG="${4:-}"
ROOT="$(dirname "$(realpath "$0")")/.."

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir "$WORK/MeshLibRust"
cp "$ROOT/source/MeshLibRust/Cargo.toml" "$ROOT/source/MeshLibRust/build.rs" "$WORK/MeshLibRust/"
cp -r "$SRC_DIR" "$WORK/MeshLibRust/src"

# The crate version is the MeshLib version. Cargo wants semver, so the fourth number becomes build metadata: 3.1.4.528 -> 3.1.4+528.
CRATE_VERSION="$(echo "$VERSION" | sed -E 's/^([0-9]+\.[0-9]+\.[0-9]+)\.([0-9]+)$/\1+\2/')"
sed -i "s/^version = .*/version = \"$CRATE_VERSION\"/" "$WORK/MeshLibRust/Cargo.toml"
grep '^version' "$WORK/MeshLibRust/Cargo.toml"

if [ -n "$NUPKG" ]; then
  unzip -q "$NUPKG" -d "$WORK/nupkg"
  # NuGet runtime identifier -> `<CARGO_CFG_TARGET_OS>-<CARGO_CFG_TARGET_ARCH>`, see `build.rs`.
  for PAIR in win-x64:windows-x86_64 win-arm64:windows-aarch64 linux-x64:linux-x86_64 linux-arm64:linux-aarch64 osx-x64:macos-x86_64 osx-arm64:macos-aarch64; do
    RID="${PAIR%%:*}"
    mkdir -p "$WORK/MeshLibRust/native"
    mv "$WORK/nupkg/runtimes/$RID/native" "$WORK/MeshLibRust/native/${PAIR#*:}"
  done
  cp "$ROOT/LICENSE" "$ROOT/thirdparty/licenses/THIRD-PARTY-NOTICES.txt" "$WORK/MeshLibRust/"
fi

rm -f "$OUTPUT"
(cd "$WORK" && zip -qr "$OUTPUT" MeshLibRust)
ls -l "$OUTPUT"
