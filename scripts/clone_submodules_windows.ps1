git -C $PSScriptRoot/.. submodule update --init --depth 1 `
    thirdparty/imgui `
    thirdparty/eigen `
    thirdparty/parallel-hashmap `
    thirdparty/expected `
    thirdparty/fastmcpp `
    thirdparty/nlohmann-json `
    thirdparty/cpp-httplib `
    thirdparty/mrbind `
    thirdparty/mrbind-pybind11 `

git -C $PSScriptRoot/../thirdparty/mrbind submodule update --init --depth 1 deps/cppdecl
