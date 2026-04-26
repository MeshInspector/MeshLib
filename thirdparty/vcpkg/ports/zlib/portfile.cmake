# MeshLib overlay: alias `zlib` to vcpkg's stock `zlib-ng` port built in
# compat mode (see vcpkg.json's dependency, plus `set(ZLIB_COMPAT ON)`
# in our triplets). The real artefacts come from the dependency; this
# port itself installs nothing.
set(VCPKG_POLICY_EMPTY_PACKAGE enabled)
