set VCPKG_DEFAULT_TRIPLET=x64-windows-meshrus
vcpkg install "@%~dp0\..\requirements\windows.txt" --recurse --binarysource=clear
