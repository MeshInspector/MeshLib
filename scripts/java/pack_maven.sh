#!/bin/bash
# Packs the Maven artifacts of `com.meshinspector:meshlib` in the Maven repository layout under `<output dir>/com/meshinspector/meshlib/<version>`:
#   meshlib-<version>.jar - the bindings, `meshlib.jar` of the Java bindings archive;
#   meshlib-<version>-natives-<os>-<arch>.jar - the libraries of one platform as the resources `meshlib/native/<os>-<arch>/`
#     with `files.txt`, which the bindings extract and load on the first use;
#   meshlib-<version>-sources.jar, meshlib-<version>-javadoc.jar (a placeholder: Maven Central requires one), meshlib-<version>.pom.
# The natives are packed for every `*native_<os>-<arch>.zip` in the given directory (the release has them for the Rust crate).
# Nothing is signed: the publishing workflow adds the signatures and checksums. Needs `jar` (a JDK) and `unzip`.
# Usage: pack_maven.sh <MeshLib version, e.g. v3.1.4.544> <MeshLibJava directory of the Java bindings archive> <directory with the native archives> <output dir>
set -euo pipefail

if [ $# -ne 4 ]; then
  echo "Usage: $0 <MeshLib version> <MeshLibJava directory> <directory with the native archives> <output dir>" >&2
  exit 1
fi
APP_VERSION="$1"
JAVA_DIR="$(realpath "$2")"
NATIVE_DIR="$(realpath "$3")"
mkdir -p "$4"
OUT_ROOT="$(realpath "$4")"
ROOT="$(dirname "$(realpath "$0")")/../.."
JAR="${JAVA_HOME:+$JAVA_HOME/bin/}jar"

# Maven compares versions part by part, so the MeshLib version is used as is: v3.1.4.544 -> 3.1.4.544.
if ! [[ "${APP_VERSION#v}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Can't make a Maven version from $APP_VERSION" >&2
  exit 1
fi
VERSION="${APP_VERSION#v}"
OUT="$OUT_ROOT/com/meshinspector/meshlib/$VERSION"
mkdir -p "$OUT"
NAME="meshlib-$VERSION"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir "$WORK/meta"
cp "$ROOT/LICENSE" "$ROOT/thirdparty/licenses/THIRD-PARTY-NOTICES.txt" "$WORK/meta/"

cp "$JAVA_DIR/meshlib.jar" "$OUT/$NAME.jar"
"$JAR" --update --file "$OUT/$NAME.jar" -C "$WORK/meta" .
"$JAR" --create --file "$OUT/$NAME-sources.jar" -C "$JAVA_DIR/src" . -C "$WORK/meta" .

mkdir "$WORK/javadoc"
cat > "$WORK/javadoc/README.txt" <<EOF
MeshLib $VERSION Java bindings.
The documentation is in the doc comments of meshlib-$VERSION-sources.jar, and at https://meshlib.io
EOF
"$JAR" --create --file "$OUT/$NAME-javadoc.jar" -C "$WORK/javadoc" .

shopt -s nullglob
PLATFORMS=()
for ZIP in "$NATIVE_DIR"/*native_*.zip; do
  PLATFORM="$(basename "$ZIP" .zip)"
  PLATFORM="${PLATFORM##*native_}"
  LIBS="$WORK/natives-$PLATFORM/meshlib/native/$PLATFORM"
  mkdir -p "$LIBS"
  unzip -q "$ZIP" -d "$LIBS"
  # The format of `sha256sum`, see `Ffi.bundledLibrary()` in the generated bindings.
  (cd "$LIBS" && sha256sum -- * > "$WORK/files.txt" && mv "$WORK/files.txt" .)
  "$JAR" --create --file "$OUT/$NAME-natives-$PLATFORM.jar" -C "$WORK/natives-$PLATFORM" . -C "$WORK/meta" .
  PLATFORMS+=("$PLATFORM")
done
if [ ${#PLATFORMS[@]} -eq 0 ]; then
  echo "No native archives in $NATIVE_DIR" >&2
  exit 1
fi

sed "s/@VERSION@/$VERSION/g" "$ROOT/scripts/java/meshlib.pom" > "$OUT/$NAME.pom"
ls -l "$OUT"
