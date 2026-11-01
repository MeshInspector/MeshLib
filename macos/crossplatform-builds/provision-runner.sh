#!/bin/bash
# Provision a self-hosted arm64 macOS runner for the native x86_64 cross build of
# MeshLib (see README.md). Idempotent. Run as the CI service account, from the repo root:
#   ./macos/crossplatform-builds/provision-runner.sh [--prewarm]
#
# The cross build compiles with the NATIVE arm64 toolchain and links the x86_64
# Homebrew at /usr/local; Rosetta runs that brew, its python and the built tests.
# Prerequisites:
#   1. Xcode Command Line Tools + a native arm64 Homebrew with cmake and ninja
#   2. Rosetta 2                                     (root -- administrator, once)
#   3. x86_64 Homebrew at /usr/local, owned by the service account (git clone; the
#      official installer is Apple-Silicon-only since 2026-09 and needs sudo)
#
# The service account has no sudo. Wherever root is needed this script prints the
# exact one-time command for an administrator and exits 1; re-run it afterwards.
# The runner must carry the labels  [self-hosted, macos, arm64, crossplatform-build].
set -euo pipefail

PREWARM=0
[[ "${1:-}" == "--prewarm" ]] && PREWARM=1

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "Run on an arm64 macOS host (cross-builds x86_64)." >&2; exit 1
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ME="$(id -un)"
X64_PREFIX=/usr/local

can_sudo() { sudo -n true 2>/dev/null; }
need_admin() {  # $1 = what is missing; the rest = command(s) for the administrator
  echo "    NEEDS ADMINISTRATOR: $1" >&2
  echo "    Run once from an admin account, then re-run this script as ${ME}:" >&2
  shift; printf '      %s\n' "$@" >&2
  exit 1
}

echo "==> 1/4  Xcode Command Line Tools (Intel formulae may build from source)"
xcode-select -p >/dev/null 2>&1 || need_admin "Command Line Tools are not installed" "xcode-select --install"
echo "    $(xcode-select -p)"

echo "==> 2/4  native arm64 toolchain (cmake + ninja)"
# Same discovery as the CI shim: the fleet has brew at /opt/homebrew or ~/.homebrew.
ARM_BREW=""
for p in /opt/homebrew "$HOME/.homebrew"; do
  [[ -x "$p/bin/brew" ]] && { ARM_BREW="$p"; break; }
done
if [[ -z "$ARM_BREW" ]]; then
  echo "    ERROR: no native Homebrew at /opt/homebrew or ~/.homebrew." >&2
  echo "    Either an administrator installs it (https://brew.sh -> /opt/homebrew), or clone" >&2
  echo "    a per-user copy:  git clone https://github.com/Homebrew/brew ~/.homebrew" >&2
  exit 1
fi
for t in cmake ninja; do
  if [[ ! -x "$ARM_BREW/bin/$t" ]]; then  # install only what's missing; never upgrade
    HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALL_UPGRADE=1 HOMEBREW_NO_ENV_HINTS=1 \
      "$ARM_BREW/bin/brew" install --quiet "$t"
  fi
  if [[ "$(file -b "$ARM_BREW/bin/$t")" == *arm64* ]]; then
    echo "    $t: $ARM_BREW (arm64) ok"
  else
    echo "    ERROR: $ARM_BREW/bin/$t is not an arm64 binary" >&2; exit 1
  fi
done

echo "==> 3/4  Rosetta 2 (runs the x86_64 brew, python and the built test binaries)"
if /usr/bin/pgrep -q oahd; then
  echo "    present"
elif can_sudo; then
  sudo softwareupdate --install-rosetta --agree-to-license
else
  need_admin "Rosetta 2 is not installed" "sudo softwareupdate --install-rosetta --agree-to-license"
fi

echo "==> 4/4  x86_64 Homebrew at ${X64_PREFIX} (source of Intel bottles)"
# Must exist and be writable: Homebrew's installer mkdir set + the git checkout dir.
X64_DIRS=(bin etc include lib sbin share var opt Cellar Caskroom Frameworks Homebrew
          share/zsh share/zsh/site-functions var/homebrew var/homebrew/linked)
# Should be writable when present (keg.rb must_be_writable_directories); `brew doctor`
# only warns about these, and formulae we use don't write there, so warn likewise.
X64_DIRS_OPT=(etc/bash_completion.d lib/cps lib/pkgconfig share/aclocal share/doc share/info
              share/locale share/man share/man/man{1..8} share/cps share/pwsh
              share/pwsh/completions var/log)
BAD=(); WARN=()
for d in "${X64_DIRS[@]}"; do
  [[ -d "${X64_PREFIX}/$d" && -w "${X64_PREFIX}/$d" ]] || BAD+=("${X64_PREFIX}/$d")
done
for d in "${X64_DIRS_OPT[@]}"; do
  [[ ! -d "${X64_PREFIX}/$d" || -w "${X64_PREFIX}/$d" ]] || WARN+=("${X64_PREFIX}/$d")
done
if (( ${#WARN[@]} )); then
  echo "    WARNING: not writable by ${ME} (brew doctor will complain): ${WARN[*]}" >&2
  echo "             fix if a formula needs it:  sudo chown -R ${ME}:staff ${WARN[*]}" >&2
fi
if (( ${#BAD[@]} )); then
  DIRS="${X64_PREFIX}/{$(IFS=,; echo "${X64_DIRS[*]}")}"
  # ${X64_PREFIX} itself stays root:wheel; only its children are handed to the account.
  need_admin "${#BAD[@]} dir(s) under ${X64_PREFIX} missing or not writable by ${ME} (first: ${BAD[0]})" \
    "sudo mkdir -p ${DIRS}" \
    "sudo chown -R ${ME}:staff ${DIRS}" \
    "sudo chmod ug=rwx ${DIRS}" \
    "sudo chmod go-w ${X64_PREFIX}/share/zsh ${X64_PREFIX}/share/zsh/site-functions"
fi
if [[ -x "${X64_PREFIX}/bin/brew" ]]; then
  echo "    present ($("${X64_PREFIX}/bin/brew" --version | head -1))"
else
  echo "    bootstrapping (git clone; no sudo needed)..."
  [[ -d "${X64_PREFIX}/Homebrew/.git" ]] || git clone https://github.com/Homebrew/brew "${X64_PREFIX}/Homebrew"
  ln -sfn ../Homebrew/bin/brew "${X64_PREFIX}/bin/brew"
  HOMEBREW_NO_ENV_HINTS=1 "${X64_PREFIX}/bin/brew" update --force --quiet
  chmod -R go-w "${X64_PREFIX}/share/zsh"
fi
# No `arch -x86_64` anywhere: brew at /usr/local picks its x86_64 Ruby by prefix and
# serves Intel bottles whatever the caller's arch -- exactly how the CI shim calls it.
if "${X64_PREFIX}/bin/brew" config 2>/dev/null | grep -q 'macOS:.*x86_64'; then
  echo "    reports an x86_64 (Intel) platform ok"
else
  echo "    WARNING: ${X64_PREFIX} brew does not report an x86_64 platform." >&2
fi

if (( PREWARM )); then
  echo "==> x86_64 formulae pre-warm (optional; CI installs these anyway)"
  # Intel macOS is Homebrew Tier 3 (no new bottles): a fresh formula may build from
  # source. Keep the runner pinned afterwards -- never `brew upgrade` this prefix.
  CLANG_VER="$(xargs < "$REPO_ROOT/scripts/mrbind/clang_version_macos.txt")"
  { cat "$REPO_ROOT/requirements/macos.txt"; printf '%s\n' pybind11 make grep lld "llvm@${CLANG_VER}"; } \
    | HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALL_UPGRADE=1 HOMEBREW_NO_ENV_HINTS=1 \
      xargs "${X64_PREFIX}/bin/brew" install --quiet
else
  echo "==> formulae pre-warm skipped (pass --prewarm to install them now)"
fi

echo "==> done. See macos/crossplatform-builds/README.md"
