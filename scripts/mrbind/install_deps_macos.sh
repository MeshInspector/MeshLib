#!/bin/bash

# Installs everything needed to generate and build MRBind bindings.
# We assume `brew` is already installed. Automatic its installation is too much,
#   especially because of the conflicts that happen if several users install it.

brew update
brew install --quiet make grep

# Nothing LLVM is installed from brew: the compiler and the linker both come from the
# llvm-pgo keg of https://github.com/MeshInspector/toolchains. Homebrew has no `lld`
# bottle for Intel macOS since it became a Tier 3 configuration, and its `llvm@N` builds
# clang;clang-tools-extra;mlir;polly, so it contains no ld64.lld to fall back on either.
if [[ -z "${LLVM_PREFIX}" ]] ; then
  echo "LLVM_PREFIX is unset: install the llvm-pgo keg (see the toolchains repo) and" >&2
  echo "point LLVM_PREFIX at it, e.g. \$(brew --prefix)/Cellar/llvm-pgo/22.1.8_2" >&2
  exit 1
fi

if [[ ! -x "${LLVM_PREFIX}/bin/ld64.lld" ]] ; then
  echo "no ld64.lld in ${LLVM_PREFIX}/bin: the bindings are linked with lld, install it" >&2
  echo "from the lld-22.1.8-pgo-macos release of the toolchains repo" >&2
  exit 1
fi
