#!/bin/bash
# Provision a self-hosted arm64 macOS runner for the NATIVE (non-Rosetta) x86_64
# cross build of MeshLib (see README.md). Idempotent -- safe to re-run.
# Run once per runner, from the repo root:
#   ./macos/crossplatform-builds/provision-runner.sh [--prewarm]
#
# The native cross build uses a NATIVE arm64 toolchain (fast compiles) that
# cross-targets x86_64 and links the x86_64 Homebrew at /usr/local.
#
# Prerequisites:
#   1. native arm64 cmake + ninja (/opt/homebrew)  -- the build driver (runs natively)
#   2. x86_64 Homebrew at /usr/local (+ formulae)   -- the x86_64 libraries to link against
#   3. Rosetta 2                                     -- ONLY to run the resulting x86_64 test
#                                                       binaries and CMake configure-time
#                                                       probes; compilation itself is native.
#
# The runner must also carry the labels  [self-hosted, macos, arm64, build].
set -euo pipefail

PREWARM=0
[[ "${1:-}" == "--prewarm" ]] && PREWARM=1

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "Run on an arm64 macOS host (cross-builds x86_64)." >&2; exit 1
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "==> 1/3  native arm64 toolchain (cmake + ninja @ /opt/homebrew)"
if [[ ! -x /opt/homebrew/bin/brew ]]; then
  echo "    ERROR: native arm64 Homebrew not found at /opt/homebrew." >&2
  echo "    Install it first: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"" >&2
  exit 1
fi
# Install only what's missing -- don't force-upgrade the runner's existing tools.
for t in cmake ninja; do
  if [[ ! -x "/opt/homebrew/bin/$t" ]]; then
    HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_ENV_HINTS=1 /opt/homebrew/bin/brew install --quiet "$t"
  fi
  case "$(file -b "/opt/homebrew/bin/$t" 2>/dev/null)" in
    *arm64*) echo "    $t: arm64 (native) ok" ;;
    *) echo "    WARNING: /opt/homebrew/bin/$t missing or not arm64" >&2 ;;
  esac
done

echo "==> 2/3  Rosetta 2 (to run x86_64 output + configure probes)"
if /usr/bin/pgrep -q oahd; then
  echo "    already installed"
else
  softwareupdate --install-rosetta --agree-to-license
fi

echo "==> 3/3  x86_64 Homebrew at /usr/local (source of x86_64 bottles)"
if [[ -x /usr/local/bin/brew ]]; then
  echo "    already present ($(arch -x86_64 /usr/local/bin/brew --version | head -1))"
else
  echo "    bootstrapping x86_64 Homebrew (will prompt for sudo)..."
  arch -x86_64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi
if ! arch -x86_64 /usr/local/bin/brew config 2>/dev/null | grep -qi 'macOS:.*x86_64'; then
  echo "    WARNING: /usr/local brew does not report an x86_64 platform." >&2
fi

if [[ "$PREWARM" == "1" ]]; then
  echo "==> x86_64 formulae pre-warm (optional; CI installs these anyway)"
  CLANG_VER="$(xargs < "$REPO_ROOT/scripts/mrbind/clang_version_macos.txt")"
  { cat "$REPO_ROOT/requirements/macos.txt"; printf '%s\n' pybind11 make grep lld "llvm@${CLANG_VER}"; } \
    | HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_ENV_HINTS=1 \
      xargs arch -x86_64 /usr/local/bin/brew install --quiet
else
  echo "==> formulae pre-warm skipped (pass --prewarm to install them now)"
fi

echo "==> done. Native x86_64 cross build recipe: macos/crossplatform-builds/README.md"
