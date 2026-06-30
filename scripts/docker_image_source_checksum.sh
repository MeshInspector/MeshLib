#!/usr/bin/env bash
# Print the content-addressed tag (source-checksum-<hash>) for a Linux Docker
# image, hashing only the sources that image is built from. Same inputs ->
# same tag, so prepare-images can reuse an image already in the registry
# instead of rebuilding it.
# Usage: docker_image_source_checksum.sh <distro>
set -euo pipefail

distro=$1
checksum() { git hash-object --stdin | cut -c1-16; }

case "${distro}" in
  ubuntu22|ubuntu24|emscripten|emscripten-build-c-bindings)
    # The thirdparty-building images: own Dockerfile + the bundled thirdparty
    # sources and build scripts. Excludes mirror the linux-changes filter
    # (the Dockerfiles also COPY source/.git, deliberately not hashed so app
    # code changes don't churn the base images).
    hash=$(git ls-files -s -- \
      "docker/${distro}Dockerfile" cmake scripts requirements thirdparty \
      ':(exclude)thirdparty/install.bat' \
      ':(exclude)thirdparty/vcpkg/**' \
      ':(exclude)thirdparty/mrbind' \
      ':(exclude)thirdparty/mrbind/**' \
      ':(exclude)thirdparty/Noto_Sans/**' \
      | checksum)
    ;;
  emscripten-generate-c-bindings)
    hash=$(git ls-files -s -- \
      docker/emscripten-generate-c-bindingsDockerfile \
      scripts/mrbind/install_deps_ubuntu.sh \
      scripts/mrbind/clang_version.txt \
      | checksum)
    ;;
  *)
    echo "unknown distro: ${distro}" >&2
    exit 1
    ;;
esac

echo "source-checksum-${hash}"
