#!/usr/bin/env bash
# Assembles the npm package of the generated JS bindings: `scripts/npm/<variant>` (package.json, README) plus the generated
# `source/MeshLibJS/js` (the ES modules, the declarations, and the wasm modules built next to them).
# Usage: assemble.sh <meshlib|meshlib-mt> [version tag]
set -euo pipefail

VARIANT=$1

VERSION="0.0.0"
if [ "${2:-}" ] ; then
  VERSION=$2
  VERSION=${VERSION#v}    # v1.2.3.4 -> 1.2.3.4
  VERSION=${VERSION%%-*}  # drop any -namespace
  VERSION="${VERSION%.*}-${VERSION##*.}"   # 1.2.3.4 -> 1.2.3-4
fi

JS_DIR=source/MeshLibJS/js
for f in index.mjs index.d.mts ffi.mjs MeshLibC2.mjs MeshLibC2.wasm MeshLibC2.node.mjs MeshLibC2.node.wasm ; do
  [ -f "${JS_DIR}/${f}" ] || { echo "Missing ${JS_DIR}/${f}: generate the bindings (TARGET=js) and build the MeshLibJS and MeshLibJSNode targets first" >&2 ; exit 1 ; }
done

DISTR_DIR=./npm-distr
rm -rf ${DISTR_DIR}
mkdir ${DISTR_DIR}

cp -a scripts/npm/${VARIANT} ${DISTR_DIR}/
cp -a ${JS_DIR}/index.mjs ${JS_DIR}/index.d.mts ${JS_DIR}/ffi.mjs ${JS_DIR}/ffi.d.mts ${JS_DIR}/lib ${JS_DIR}/ns ${DISTR_DIR}/${VARIANT}/
cp -a ${JS_DIR}/MeshLibC2.mjs ${JS_DIR}/MeshLibC2.wasm ${JS_DIR}/MeshLibC2.node.mjs ${JS_DIR}/MeshLibC2.node.wasm ${DISTR_DIR}/${VARIANT}/
cp LICENSE ${DISTR_DIR}/${VARIANT}/

pushd ${DISTR_DIR}/${VARIANT}/
  npm version "${VERSION}" --no-git-tag-version --allow-same-version >/dev/null
popd
