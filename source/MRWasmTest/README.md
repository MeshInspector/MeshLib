# MRWasmTest

Headless Node tests for the MeshLib WebAssembly bindings ([`source/MRWasmModule`](../MRWasmModule)).
`roundtrip.test.mjs` asserts the import &rarr; boolean &rarr; decimate &rarr; export round-trip, with
no browser or Three.js dependency.

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
MESHLIB_MODULE=build/Release/bin/meshlib.mjs node source/MRWasmTest/roundtrip.test.mjs
```

Or copy `meshlib.mjs` + `meshlib.wasm` into this directory and run `npm test`. The test prints `OK`
and exits 0 on success.
