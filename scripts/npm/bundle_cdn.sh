#!/usr/bin/env bash
# Bundles the generated modules of a meshlib or meshlib-mt package (`index.mjs`, `ffi.mjs`, `lib/`, `ns/`) into one ES module
# for the CDN (`publish-wasm.yml`, tested by `build-test-emscripten.yml`): whitespace and syntax minified, the identifiers kept
# for readable stack traces. The Emscripten glue stays a separate file, imported from the bundle as `./MeshLibC2.mjs`
# (`./MeshLibC2.node.mjs` under Node), so it must be next to the bundle with its wasm.
# Usage: bundle_cdn.sh <package directory> <output .mjs>
set -euo pipefail

npx --yes esbuild@0.25.12 "$1/index.mjs" --bundle --format=esm --platform=neutral --minify-whitespace --minify-syntax \
  --external:./MeshLibC2.mjs --external:./MeshLibC2.node.mjs --log-level=warning --outfile="$2"
