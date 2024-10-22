#!/bin/bash

FRAMEWORK_PATH="./Library/Frameworks/MeshLib.framework/Versions/Current"
ENTITLEMENTS_PATH="./macos/entitlements.plist"

DEVELOPER_ID_APPLICATION="Developer ID Application: ASGSoft LLC (465Q5Z6W45)"
DEVELOPER_ID_INSTALLER="Developer ID Installer: ASGSoft LLC (465Q5Z6W45)"

SKIP_SIGN=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --skip-sign) SKIP_SIGN=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

sign_file() {
    local file="$1"
    echo "Code signing file: ${file}"
    codesign --sign "${DEVELOPER_ID_APPLICATION}" --entitlements "${ENTITLEMENTS_PATH}" --verbose --timestamp --options=runtime --force "${file}"
}

if [ "$SKIP_SIGN" = false ]; then
    echo "Signing the main binaries MeshViewer and meshconv"
    sign_file "${FRAMEWORK_PATH}/bin/MeshViewer"
    sign_file "${FRAMEWORK_PATH}/bin/meshconv"

    echo "Verifying the signatures for MeshViewer and meshconv"
    codesign --verify --verbose=2 "${FRAMEWORK_PATH}/MeshLib/MeshViewer"
    codesign --verify --verbose=2 "${FRAMEWORK_PATH}/MeshLib/meshconv"

    echo "Signing additional libraries and components"
    find "${FRAMEWORK_PATH}/libs" -type f \( -name "*.dylib" -o -name "*.so" -o -name "*.bundle" -o -name "*.plugin" \) | while read -r FILE; do
        sign_file "${FILE}"
    done

    echo "Signing the entire framework directory"
    codesign --sign "${DEVELOPER_ID_APPLICATION}" --entitlements "${ENTITLEMENTS_PATH}" --verbose --deep  "./MeshLib.framework"

    echo "Verifying code signature for the framework"
    codesign --verify --deep --strict --verbose=2 "./MeshLib.framework"

    echo "Displaying code signing information for the framework"
    codesign -dvv --verbose=4 "./MeshLib.framework"
else
    echo "--skip-sign option detected, skipping the signing process."
fi

echo "Building package from .framework"
if [ "$SKIP_SIGN" = false ]; then
    pkgbuild \
        --root Library \
        --identifier com.MeshInspector.MeshLib \
        --sign "${DEVELOPER_ID_INSTALLER}" \
        --install-location /Library \
        MeshLib.pkg

    productbuild \
        --distribution ./macos/Distribution.xml \
        --package-path ./MeshLib.pkg \
        --resources ./macos/Resources \
        --sign "${DEVELOPER_ID_INSTALLER}" \
        MeshLib_signed.pkg
else
    pkgbuild \
        --root Library \
        --identifier com.MeshInspector.MeshLib \
        --install-location /Library \
        MeshLib.pkg

    productbuild \
        --distribution ./macos/Distribution.xml \
        --package-path ./MeshLib.pkg \
        --resources ./macos/Resources \
        MeshLib_unsigned.pkg
fi

echo "Verifying package signature"
if [ "$SKIP_SIGN" = false ]; then
    spctl -a -t install -vvv --context context:primary-signature MeshLib.pkg || true
    mv MeshLib_signed.pkg meshlib.pkg
else
    echo "Skipping signature verification because the package was not signed."
    mv MeshLib_unsigned.pkg meshlib.pkg
fi
