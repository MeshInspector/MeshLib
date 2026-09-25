#!/bin/sh
set -e

case "$(uname -m)" in
    x86_64|amd64)  arch=x64 ;;
    arm64|aarch64) arch=arm64 ;;
esac

case "$(uname -s)" in
    Linux)  os=linux ;;
    Darwin) os=osx ;;
esac

echo "${arch}-${os}-meshlib"
