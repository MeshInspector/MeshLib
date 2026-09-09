Use MeshLib as a WebAssembly geometry backend for a Three.js app: convert a
`BufferGeometry` to a MeshLib mesh, run boolean (CSG) and decimate, and convert the
result back.

## Build and run

From the MeshLib root, in an Emscripten environment, build the module and copy it here:

```sh
MESHLIB_WASM_MODULE_TARGET=web ./scripts/build_wasm_meshlib.sh
cp build/Release/bin/meshlib.{mjs,wasm} examples/threejs/
python3 -m http.server -d examples/threejs 8080
```
