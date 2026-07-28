# MRWasmTest

Headless Node tests for the MeshLib WebAssembly bindings ([`source/MRWasmModule`](../MRWasmModule)).

- `index.mjs` — the main test set. No disk IO, so it runs against any build, including the
  browser `web`-target module (no `NODERAWFS`).
- `io.test.mjs` — file round-trips (mesh / point-cloud / voxel). These need a real filesystem, so
  they only run against a `NODERAWFS` (node-target) module and are not part of `index.mjs`.

## Build the bindings

From the MeshLib root, inside an Emscripten environment (the same `emscripten/emsdk` image the rest
of MeshLib's Wasm uses):

```sh
./scripts/build_wasm_meshlib.sh
# -> build/Release/bin/meshlib.mjs and build/Release/bin/meshlib.wasm
```

## Run the tests

Point `MESHLIB_MODULE` at the built module (its `meshlib.wasm` is loaded from the same directory):

```sh
# from the MeshLib root
MESHLIB_MODULE=build/Release/bin/meshlib.mjs node source/MRWasmTest/index.mjs     # main set
MESHLIB_MODULE=build/Release/bin/meshlib.mjs node source/MRWasmTest/io.test.mjs   # + disk IO
```

Or copy `meshlib.mjs` + `meshlib.wasm` into this directory and run `npm test` (which runs both).
Each entry prints `OK` / `IO OK` and exits 0 on success.
