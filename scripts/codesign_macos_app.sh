#!/bin/bash

APP_PATH="./MeshLib.framework/Versions/Current"
ENTITLEMENTS_PATH="./macos/entitlements.plist"

DEVELOPER_ID_APPLICATION="Developer ID Application: ASGSoft LLC (465Q5Z6W45)"
DEVELOPER_ID_INSTALLER="Developer ID Installer: ASGSoft LLC (465Q5Z6W45)"

sign_file() {
    local file="$1"
    echo "Code signing file: ${file}"
    codesign --sign "${DEVELOPER_ID_APPLICATION}" --entitlements "${ENTITLEMENTS_PATH}" --verbose --timestamp --options=runtime --force "${file}"
}

# Sign each of the main binaries individually
echo "Signing the main binaries MeshViewer and meshconv"
sign_file "${APP_PATH}/bin/MeshViewer"
sign_file "${APP_PATH}/bin/meshconv"

# Verify each binary signature
echo "Verifying the signatures for MeshViewer and meshconv"
codesign --verify --verbose=2 "${APP_PATH}/MeshLib/MeshViewer"
codesign --verify --verbose=2 "${APP_PATH}/MeshLib/meshconv"

# Sign other components like dynamic libraries, plugins, etc.
echo "Signing additional libraries and components"
find "${APP_PATH}/libs" -type f \( -name "*.dylib" -o -name "*.so" -o -name "*.bundle" -o -name "*.plugin" \) | while read -r FILE; do
    sign_file "${FILE}"
done

# Now sign the entire framework directory
echo "Signing the entire framework directory"
codesign --sign "${DEVELOPER_ID_APPLICATION}" --entitlements "${ENTITLEMENTS_PATH}" --verbose --deep  "./MeshLib.framework"

# Verifying the framework
echo "Verifying code signature for the framework"
codesign --verify --deep --strict --verbose=2 "./MeshLib.framework"

# Display detailed code signing information
echo "Displaying code signing information for the framework"
codesign -dvv --verbose=4 "./MeshLib.framework"

# Package building process
echo "Building package from .framework"
pkgbuild \
            --root MeshLib.framework \
            --identifier com.MeshInspector.MeshLib \
            --sign "${DEVELOPER_ID_INSTALLER}" \
            --install-location /Library/Frameworks/MeshLib.framework \
            MeshLib.pkg


productbuild \
          --distribution ./macos/Distribution.xml \
          --package-path ./MeshLib.pkg \
          --resources ./macos/Resources \
          --sign "${DEVELOPER_ID_INSTALLER}" \
          MeshLib_signed.pkg

# Verify package code signature
echo "Verifying package signature"
spctl -a -t install -vvv --context context:primary-signature MeshLib_signed.pkg || true
