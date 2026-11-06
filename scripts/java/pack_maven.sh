#!/bin/bash
# Packs the Maven artifacts of `com.meshinspector:meshlib` in the Maven repository layout under `<output dir>/com/meshinspector/meshlib/<version>`:
#   meshlib-<version>.jar - the bindings, `meshlib.jar` of the Java bindings archive, plus the resource `meshlib/native/native.txt`:
#     the release download URL and the SHA-256 of every `*native_<os>-<arch>.zip` in the given directory (the release has them for the Rust crate),
#     from which the bindings download the libraries of the current platform on the first use;
#   meshlib-<version>-sources.jar, meshlib-<version>-javadoc.jar (a placeholder: Maven Central requires one), meshlib-<version>.pom.
# Nothing is signed: the publishing workflow adds the signatures and checksums. Needs `jar` (a JDK).
# Usage: pack_maven.sh <MeshLib version, e.g. v1.2.3.4> <MeshLibJava directory of the Java bindings archive> <directory with the native archives> <release download URL> <output dir>
set -euo pipefail

if [ $# -ne 5 ]; then
  echo "Usage: $0 <MeshLib version> <MeshLibJava directory> <directory with the native archives> <release download URL> <output dir>" >&2
  exit 1
fi
APP_VERSION="$1"
JAVA_DIR="$(realpath "$2")"
NATIVE_DIR="$(realpath "$3")"
URL="$4"
mkdir -p "$5"
OUT_ROOT="$(realpath "$5")"
ROOT="$(dirname "$(realpath "$0")")/../.."
JAR="${JAVA_HOME:+$JAVA_HOME/bin/}jar"

# Maven compares versions part by part, so the MeshLib version is used as is: v1.2.3.4 -> 1.2.3.4.
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

# The format of the Rust crate's `native.txt`, see `downloadNativeLibraries()` in the generated `Ffi`.
mkdir -p "$WORK/native/meshlib/native"
NATIVE_TXT="$WORK/native/meshlib/native/native.txt"
echo "url $URL" > "$NATIVE_TXT"
shopt -s nullglob
COUNT=0
for ZIP in "$NATIVE_DIR"/*native_*.zip; do
  echo "$(sha256sum "$ZIP" | cut -d' ' -f1) $(basename "$ZIP")" >> "$NATIVE_TXT"
  COUNT=$((COUNT + 1))
done
if [ $COUNT -eq 0 ]; then
  echo "No native archives in $NATIVE_DIR" >&2
  exit 1
fi
cat "$NATIVE_TXT"

cp "$JAVA_DIR/meshlib.jar" "$OUT/$NAME.jar"
"$JAR" --update --file "$OUT/$NAME.jar" -C "$WORK/meta" . -C "$WORK/native" .
"$JAR" --create --file "$OUT/$NAME-sources.jar" -C "$JAVA_DIR/src" . -C "$WORK/meta" .

mkdir "$WORK/javadoc"
cat > "$WORK/javadoc/README.txt" <<EOF
MeshLib $VERSION Java bindings.
The documentation is in the doc comments of meshlib-$VERSION-sources.jar, and at https://meshlib.io
EOF
"$JAR" --create --file "$OUT/$NAME-javadoc.jar" -C "$WORK/javadoc" .

sed "s/@VERSION@/$VERSION/g" "$ROOT/scripts/java/meshlib.pom" > "$OUT/$NAME.pom"
ls -l "$OUT"
