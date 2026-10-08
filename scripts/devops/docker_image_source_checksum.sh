#!/usr/bin/env bash
# Usage: docker_image_source_checksum.sh <distro>
set -euo pipefail

distro=$1

emscripten=(
  scripts/build_thirdparty.sh
  scripts/ask_emscripten_mode.src
  scripts/thirdparty
  thirdparty
  ':(exclude)thirdparty/install.bat'
  ':(exclude)thirdparty/vcpkg/**'
  ':(exclude)thirdparty/mrbind'
  ':(exclude)thirdparty/mrbind/**'
  ':(exclude)thirdparty/Noto_Sans/**'
  # license texts shipped in packages, not an image input: including them would
  # move the source-checksum-* tag on licenses-only commits and force a needless
  # rebuild of every image (the registry-check finds nothing at the new tag).
  ':(exclude)thirdparty/licenses/**'
  cmake/Modules/ConfigureVcpkg.cmake
  cmake/Modules/DefaultEmscriptenOptions.cmake
  scripts/cmake_install.sh
)

ubuntu=(
  scripts/build_cpm_thirdparty.sh
  scripts/ask_emscripten_mode.src
  thirdparty/cpm
  cmake/Modules/ConfigureVcpkg.cmake
  requirements/ubuntu.txt
  requirements/python/requirements.txt
  scripts/install_apt_requirements.sh
)

case "${distro}" in
  ubuntu22|ubuntu24|ubuntu26)
    files=( "docker/${distro}Dockerfile" "${ubuntu[@]}" ) ;;
  emscripten|emscripten-build-c-bindings)
    files=( "docker/${distro}Dockerfile" "${emscripten[@]}" ) ;;
  emscripten-generate-c-bindings)
    files=( "docker/${distro}Dockerfile" ) ;;
  rockylinux8-vcpkg|rockylinux9-vcpkg)
    files=( docker/rockylinux8-vcpkgDockerfile docker/rockylinux9-vcpkgDockerfile thirdparty/vcpkg ) ;;
  *)
    echo "unknown distro: ${distro}" >&2
    exit 1 ;;
esac

echo "source-checksum-$(git ls-files -s -- "${files[@]}" | git hash-object --stdin | cut -c1-16)"
