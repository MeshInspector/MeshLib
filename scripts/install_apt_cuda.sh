#!/bin/bash
set -eo pipefail

source /etc/os-release
case "$NAME" in 
    "Ubuntu")
        DISTRO="ubuntu${VERSION_ID/./}" ;;
    *)
        echo "Unsupported distro: $NAME"
        exit 1 ;;
esac

case "$(uname -m)" in
    "x86_64")
        ARCH="x86_64" ;;
    "aarch64")
        ARCH="sbsa" ;;
    *)
        echo "Unsupported arch: $(uname -m)"
        exit 1 ;;
esac

pushd $(mktemp -d)
wget "https://developer.download.nvidia.com/compute/cuda/repos/$DISTRO/$ARCH/cuda-keyring_1.1-1_all.deb"
dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb
popd

case "$DISTRO" in
    "ubuntu2004")
        CUDA_VERSION="12-1" ;;
    # the ubuntu2604 repo starts at 13.3
    "ubuntu2604")
        CUDA_VERSION="13-3" ;;
    *)
        CUDA_VERSION="12-6" ;;
esac

apt update
apt install -y "cuda-minimal-build-$CUDA_VERSION"
