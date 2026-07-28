#!/usr/bin/env bash
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
cp build/Release/bin/${VARIANT}{,.node}.{mjs,wasm} ${DISTR_DIR}/${VARIANT}/
cp build/Release/bin/${VARIANT}.d.mts ${DISTR_DIR}/${VARIANT}/bindings.d.mts
cp scripts/npm/index.d.mts ${DISTR_DIR}/${VARIANT}/
cp LICENSE ${DISTR_DIR}/${VARIANT}/

pushd ${DISTR_DIR}/${VARIANT}/
  npm version "${VERSION}" --no-git-tag-version --allow-same-version >/dev/null
popd
