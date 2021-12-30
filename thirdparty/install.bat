set VCPKG_DEFAULT_TRIPLET=x64-windows-meshrus
vcpkg install gtest tl-expected eigen3 boost-dynamic-bitset boost-integer boost-move boost-functional boost-range openvdb freetype gdcm jsoncpp libzip cpr boost-geometry spdlog suitesparse boost-signals2 python3 tiff podofo --recurse --binarysource=clear
