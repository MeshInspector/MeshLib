#!/usr/bin/env bash
# Usage: docker_image_source_checksum.sh <distro>
set -euo pipefail

distro=$1

common=(
  scripts/build_thirdparty.sh
  scripts/ask_emscripten_mode.src
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
)

ubuntu=(
  requirements/ubuntu.txt
  requirements/python/requirements.txt
  scripts/install_apt_requirements.sh
  scripts/install_thirdparty.sh
  scripts/mrbind-pybind11/python_versions.txt
  scripts/mrbind-pybind11/install_all_python_versions_ubuntu_pkgs.sh
)
emscripten=( scripts/cmake_install.sh )

case "${distro}" in
  ubuntu22|ubuntu24|ubuntu26)
    files=( "docker/${distro}Dockerfile" "${common[@]}" "${ubuntu[@]}" ) ;;
  emscripten|emscripten-build-c-bindings)
    files=( "docker/${distro}Dockerfile" "${common[@]}" "${emscripten[@]}" ) ;;
  emscripten-generate-c-bindings)
    files=( "docker/${distro}Dockerfile" ) ;;
  rockylinux8-vcpkg|rockylinux9-vcpkg)
    files=( docker/rockylinux8-vcpkgDockerfile docker/rockylinux9-vcpkgDockerfile thirdparty/vcpkg ) ;;
  *)
    echo "unknown distro: ${distro}" >&2
    exit 1 ;;
esac

echo "source-checksum-$(git ls-files -s -- "${files[@]}" | git hash-object --stdin | cut -c1-16)"
