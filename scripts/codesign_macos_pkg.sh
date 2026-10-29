#!/bin/bash
# Re-signs a MeshLib .pkg with the Developer ID certificates: every Mach-O file of the payload with
# Developer ID Application and the hardened runtime (required for notarization), then the package
# with Developer ID Installer. The certificates must be in the keychain search list.
# usage: codesign_macos_pkg.sh <unsigned.pkg> <signed.pkg>

set -euo pipefail

IN_PKG="$1"
OUT_PKG="$2"

DEVELOPER_ID_APPLICATION="Developer ID Application: AMV Consulting LLC (3KYM6LJYCG)"
DEVELOPER_ID_INSTALLER="Developer ID Installer: AMV Consulting LLC (3KYM6LJYCG)"

MACOS_DIR="$(cd "$(dirname "$0")/../macos" && pwd)"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

pkgutil --expand-full "${IN_PKG}" "${WORK_DIR}/expanded"
PAYLOAD="${WORK_DIR}/expanded/MeshLib.pkg/Payload"
[ -L "${PAYLOAD}/Frameworks/MeshLib.framework/Versions/Current" ]

# static libraries and object files are not signed, and notarization does not require it
find "${PAYLOAD}" -type f -print0 | while IFS= read -r -d '' FILE
do
    TYPE="$(file -b "${FILE}")"
    case "${TYPE}" in
        Mach-O*object*|Mach-O*"ar archive"*) ;;
        Mach-O*executable*)
            codesign --force --timestamp --options runtime --entitlements "${MACOS_DIR}/entitlements.plist" \
                --sign "${DEVELOPER_ID_APPLICATION}" "${FILE}" ;;
        Mach-O*)
            codesign --force --timestamp --options runtime --sign "${DEVELOPER_ID_APPLICATION}" "${FILE}" ;;
    esac
done

# otherwise pkgbuild adds a hidden ._<name> companion file for every file in the package
xattr -cr "${PAYLOAD}"

# same arguments as in distribution_apple.sh
pkgbuild \
  --root "${PAYLOAD}" \
  --identifier com.MeshInspector.MeshLib \
  --install-location /Library \
  --sign "${DEVELOPER_ID_INSTALLER}" \
  "${WORK_DIR}/MeshLib.pkg"

productbuild \
  --distribution "${MACOS_DIR}/Distribution.xml" \
  --package-path "${WORK_DIR}/MeshLib.pkg" \
  --resources "${WORK_DIR}/expanded/Resources" \
  --sign "${DEVELOPER_ID_INSTALLER}" \
  "${OUT_PKG}"

pkgutil --check-signature "${OUT_PKG}"
