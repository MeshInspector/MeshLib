# MeshLib &times; Three.js

Use MeshLib as a geometry-processing backend for a Three.js app: import a
`BufferGeometry`, run boolean (CSG) and decimate in WebAssembly, and export the
result back to a `BufferGeometry`.

This directory contains:

| File | What it is |
| --- | --- |
| `meshlib-three.js` | Thin adapter between `THREE.BufferGeometry` and the MeshLib module (THREE is injected; no bundled dependency). |
| `index.html` | Interactive browser demo: box &cup; sphere &rarr; decimate &rarr; render. |
| `roundtrip.test.mjs` | Headless Node test (no browser/THREE) asserting import &rarr; boolean &rarr; decimate &rarr; export correctness. |
| `package.json` | Marks the directory as an ES module package and defines `npm test`. |

The module itself (`meshlib.mjs` + `meshlib.wasm`) is **built**, not checked in.

## Public API

```js
import createMeshLib from './meshlib.mjs';
const ml = await createMeshLib();

const a   = ml.meshFromGeometry(positionsFloat32, indicesUint32); // -> Mesh
const b   = ml.meshFromGeometry(/* ... */);
const out = ml.boolean(a, b, ml.BooleanOp.Union);                 // Union | Intersection | DifferenceAB | DifferenceBA
const res = ml.decimate(out, { targetRatio: 0.5 });               // mutates `out`; -> { vertsDeleted, facesDeleted, errorIntroduced, cancelled }

const geo = out.toGeometry();             // { positions: Float32Array, indices: Uint32Array }
const geoN = out.toGeometryWithNormals(); // + normals: Float32Array

a.delete(); b.delete(); out.delete();     // free the C++ meshes (embind handles)
```

`decimate` accepts any of: `targetRatio` (fraction of triangles to keep, 0..1),
`targetTriangleCount`, `maxDeletedFaces`, `maxError`, `maxEdgeLen`,
`strategy` (`'minimizeError'` | `'shortestEdgeFirst'`). At least one stopping
criterion is required.

## Build the module

The module is built with the Emscripten toolchain (the same `emscripten/emsdk`
image the rest of MeshLib's Wasm uses). It links only `MRMesh` — no viewer — and
is single-threaded by default so it runs on any page without COOP/COEP
cross-origin isolation.

```sh
# from the repo root, inside an Emscripten environment with thirdparty prebuilt
./scripts/build_wasm_meshlib.sh
# -> build/Release/bin/meshlib.mjs and build/Release/bin/meshlib.wasm
```

Copy both files next to these examples:

```sh
cp build/Release/bin/meshlib.mjs build/Release/bin/meshlib.wasm examples/threejs/
```

## Run the browser demo

ES modules must be served over HTTP (not `file://`):

```sh
cd examples/threejs
python3 -m http.server 8080
# open http://localhost:8080/
```

## Run the headless test

```sh
cd examples/threejs
node roundtrip.test.mjs           # expects meshlib.mjs/.wasm in this dir
# or point it elsewhere:
MESHLIB_MODULE=/path/to/meshlib.mjs node roundtrip.test.mjs
```

It prints `OK` and exits 0 on success.
