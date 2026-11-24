#!/usr/bin/env bash
# Assembles an npm package in `./npm-distr/<variant>`: `scripts/npm/<variant>` (package.json, README) plus the built module.
# - meshlib, meshlib-mt: the generated JS bindings, copied to the build tree with the wasm modules built next to it
#   (`build/Release/MeshLibJS/js` by default, `JS_DIR` overrides).
# - meshlib-light, meshlib-light-mt: the handwritten Embind module from `build/Release/bin`, its `.d.mts` documented by `inject_tsdoc.py` first.
# Usage: assemble.sh <meshlib|meshlib-mt|meshlib-light|meshlib-light-mt> [version tag]
set -euo pipefail

VARIANT=$1

VERSION="0.0.0"
if [ "${2:-}" ] ; then
  VERSION=$2
  VERSION=${VERSION#v}    # v1.2.3.4 -> 1.2.3.4
  VERSION=${VERSION%%-*}  # drop any -namespace
  VERSION="${VERSION%.*}-${VERSION##*.}"   # 1.2.3.4 -> 1.2.3-4
fi

DISTR_DIR=./npm-distr
rm -rf ${DISTR_DIR}
mkdir ${DISTR_DIR}

cp -a scripts/npm/${VARIANT} ${DISTR_DIR}/
case "${VARIANT}" in
  meshlib|meshlib-mt)
    JS_DIR=${JS_DIR:-build/Release/MeshLibJS/js}
    for f in index.mjs index.d.mts ffi.mjs MeshLibC2.mjs MeshLibC2.wasm MeshLibC2.node.mjs MeshLibC2.node.wasm ; do
      [ -f "${JS_DIR}/${f}" ] || { echo "Missing ${JS_DIR}/${f}: generate the bindings (TARGET=js) and build the MeshLibJS and MeshLibJSNode targets first" >&2 ; exit 1 ; }
    done
    cp -a ${JS_DIR}/index.mjs ${JS_DIR}/index.d.mts ${JS_DIR}/ffi.mjs ${JS_DIR}/ffi.d.mts ${JS_DIR}/lib ${JS_DIR}/ns ${DISTR_DIR}/${VARIANT}/
    cp -a ${JS_DIR}/MeshLibC2.mjs ${JS_DIR}/MeshLibC2.wasm ${JS_DIR}/MeshLibC2.node.mjs ${JS_DIR}/MeshLibC2.node.wasm ${DISTR_DIR}/${VARIANT}/
    ;;
  meshlib-light|meshlib-light-mt)
    cp build/Release/bin/${VARIANT}{,.node}.{mjs,wasm} ${DISTR_DIR}/${VARIANT}/
    cp build/Release/bin/${VARIANT}.d.mts ${DISTR_DIR}/${VARIANT}/bindings.d.mts
    cp scripts/npm/index.d.mts ${DISTR_DIR}/${VARIANT}/
    ;;
esac
cp LICENSE ${DISTR_DIR}/${VARIANT}/

pushd ${DISTR_DIR}/${VARIANT}/
  npm version "${VERSION}" --no-git-tag-version --allow-same-version >/dev/null
popd
