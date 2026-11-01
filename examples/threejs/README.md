Use MeshLib as a WebAssembly geometry backend for a Three.js app: convert a
`BufferGeometry` to a MeshLib mesh, run boolean (CSG) and decimate, and convert the
result back.

## Build and run

Install the package next to the demo (or build it yourself, see `source/MeshLibJS/README.md`), then serve the directory:

```sh
cd examples/threejs
npm install @meshinspector/meshlib
python3 -m http.server 8080
```
