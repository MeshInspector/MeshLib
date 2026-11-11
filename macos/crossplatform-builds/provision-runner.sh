#!/bin/bash
# Provision a self-hosted arm64 macOS runner for the native x86_64 cross build of
# MeshLib (see README.md). Idempotent. Run as the CI service account:
#   ./macos/crossplatform-builds/provision-runner.sh
#
# The service account has no sudo. Wherever root is needed this script prints the
# exact one-time command for an administrator and exits 1; re-run it afterwards.
# The runner must carry the labels  [self-hosted, macos, arm64, crossplatform-build].
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "Run on an arm64 macOS host (cross-builds x86_64)." >&2; exit 1
fi
ME="$(id -un)"
X64_PREFIX=/usr/local

need_admin() {  # $1 = what is missing; the rest = command(s) for the administrator
  echo "    NEEDS ADMINISTRATOR: $1" >&2
  echo "    Run once from an admin account, then re-run this script as ${ME}:" >&2
  shift; printf '      %s\n' "$@" >&2
  exit 1
}

echo "==> 1/4  Xcode Command Line Tools"
xcode-select -p >/dev/null 2>&1 || need_admin "Command Line Tools are not installed" "xcode-select --install"
echo "    $(xcode-select -p)"

echo "==> 2/4  native arm64 toolchain (cmake + ninja)"
# Same discovery as the CI shim: the fleet has brew at /opt/homebrew or ~/.homebrew.
ARM_BREW=""
for p in /opt/homebrew "$HOME/.homebrew"; do
  [[ -x "$p/bin/brew" ]] && { ARM_BREW="$p"; break; }
done
if [[ -z "$ARM_BREW" ]]; then
  echo "    ERROR: no native Homebrew at /opt/homebrew or ~/.homebrew. Install it" >&2
  echo "    (https://brew.sh), or clone a per-user copy:" >&2
  echo "      git clone https://github.com/Homebrew/brew ~/.homebrew" >&2
  exit 1
fi
for t in cmake ninja; do
  [[ -x "$ARM_BREW/bin/$t" ]] || HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALL_UPGRADE=1 "$ARM_BREW/bin/brew" install --quiet "$t"
  [[ "$(file -b "$ARM_BREW/bin/$t")" == *arm64* ]] || { echo "    ERROR: $ARM_BREW/bin/$t is not an arm64 binary" >&2; exit 1; }
  echo "    $t: $ARM_BREW (arm64) ok"
done

echo "==> 3/4  Rosetta 2 (runs the x86_64 brew and the built x86_64 binaries)"
/usr/bin/pgrep -q oahd || need_admin "Rosetta 2 is not installed" "sudo softwareupdate --install-rosetta --agree-to-license"
echo "    present"

echo "==> 4/4  x86_64 Homebrew at ${X64_PREFIX} (the llvm-pgo keg, GNU make and grep)"
# Homebrew's own directory set plus the git checkout dir; ${X64_PREFIX} itself stays root:wheel.
X64_DIRS=(bin etc include lib sbin share var opt Cellar Caskroom Frameworks Homebrew
          share/zsh share/zsh/site-functions var/homebrew var/homebrew/linked)
BAD=()
for d in "${X64_DIRS[@]}"; do
  [[ -d "${X64_PREFIX}/$d" && -w "${X64_PREFIX}/$d" ]] || BAD+=("${X64_PREFIX}/$d")
done
if (( ${#BAD[@]} )); then
  DIRS="${X64_PREFIX}/{$(IFS=,; echo "${X64_DIRS[*]}")}"
  need_admin "${#BAD[@]} dir(s) under ${X64_PREFIX} missing or not writable by ${ME} (first: ${BAD[0]})" \
    "sudo mkdir -p ${DIRS}" \
    "sudo chown -R ${ME}:staff ${DIRS}" \
    "sudo chmod go-w ${X64_PREFIX}/share/zsh ${X64_PREFIX}/share/zsh/site-functions"
fi
if [[ ! -x "${X64_PREFIX}/bin/brew" ]]; then
  # The official installer is Apple-Silicon-only since 2026-09 and needs sudo; a clone works.
  [[ -d "${X64_PREFIX}/Homebrew/.git" ]] || git clone https://github.com/Homebrew/brew "${X64_PREFIX}/Homebrew"
  ln -sfn ../Homebrew/bin/brew "${X64_PREFIX}/bin/brew"
  "${X64_PREFIX}/bin/brew" update --force --quiet
fi
# No `arch -x86_64` anywhere: brew at /usr/local picks its x86_64 Ruby by prefix and
# serves Intel bottles whatever the caller's arch, which is how the CI shim calls it.
"${X64_PREFIX}/bin/brew" config | grep -q 'macOS:.*x86_64' ||
  { echo "    ERROR: ${X64_PREFIX} brew does not report an x86_64 platform" >&2; exit 1; }
# What CI installs there anyway; Intel macOS is Homebrew Tier 3, so never `brew upgrade` this prefix.
HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALL_UPGRADE=1 "${X64_PREFIX}/bin/brew" install --quiet make grep zstd xz
echo "    $("${X64_PREFIX}/bin/brew" --version | head -1), x86_64 ok"

echo "==> done. The llvm-pgo keg is installed by the first CI run."
