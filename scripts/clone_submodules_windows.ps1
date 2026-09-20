git -C "$PSScriptRoot\.." submodule update --init --depth 1 `
    thirdparty/imgui `
    thirdparty/eigen `
    thirdparty/expected `
    thirdparty/fastmcpp `
    thirdparty/nlohmann-json `
    thirdparty/cpp-httplib `
    thirdparty/mrbind `
    thirdparty/mrbind-pybind11
if ( $LASTEXITCODE -ne 0 ) { exit $LASTEXITCODE }

git -C "$PSScriptRoot\..\thirdparty\mrbind" submodule update --init --depth 1 deps/cppdecl
if ( $LASTEXITCODE -ne 0 ) { exit $LASTEXITCODE }
