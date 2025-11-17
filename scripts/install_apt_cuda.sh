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
        ARCH="arm64" ;;
    *)
        echo "Unsupported arch: $(uname -m)"
        exit 1 ;;
esac

pushd $(mktemp -d)
wget "https://developer.download.nvidia.com/compute/cuda/repos/$DISTRO/$ARCH/cuda-keyring_1.1-1_all.deb"
dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb
popd

if [ $DISTRO = "ubuntu2004" ] ; then
    CUDA_VERSION="12-1"
else
    CUDA_VERSION="12-6"
fi

apt update
apt install -y "cuda-minimal-build-$CUDA_VERSION"
