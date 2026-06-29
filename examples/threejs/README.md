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

// Build a Mesh from flat typed arrays, via the native MeshLib containers:
const coords = ml.VertCoords.fromArray( positionsFloat32 );  // 3 floats per vertex
const tris   = ml.Triangulation.fromArray( indicesUint32 );  // 3 vertex ids per triangle
const a = ml.Mesh.fromTriangles( coords, tris );             // MR::Mesh::fromTriangles
coords.delete(); tris.delete();
// const b = ... built the same way from its own VertCoords + Triangulation

// Process (CSG, decimation, ...):
const res = ml.boolean( a, b, ml.BooleanOperation.Union );   // Union | Intersection | DifferenceAB | DifferenceBA
const out = res.mesh;
const s = new ml.DecimateSettings();
s.maxDeletedFaces = 1000;                                    // any DecimateSettings field
ml.decimateMesh( out, s );                                   // mutates `out`
s.delete();

// Export back to flat typed arrays (pack first: getTriangulation needs a gap-free mesh):
out.pack();
const points   = out.points;                                 // MR::Mesh::points    -> VertCoords
const topology = out.topology;                               // MR::Mesh::topology  -> MeshTopology
const tri      = topology.getTriangulation();                // MR::MeshTopology::getTriangulation
const positions = points.toArray();                          // Float32Array
const indices   = tri.toArray();                             // Uint32Array
points.delete(); topology.delete(); tri.delete();

const vn = ml.computePerVertNormals( out );                  // MR::computePerVertNormals -> VertNormals
const normals = vn.toArray();                                // Float32Array
vn.delete();

a.delete(); b.delete(); out.delete(); res.delete();
```

The JS surface mirrors MeshLib 1:1 — real classes (`Mesh`, `MeshTopology`, `VertCoords`,
`Triangulation`), the real field accessors `mesh.points` / `mesh.topology`, and real functions
(`Mesh.fromTriangles`, `MeshTopology.getTriangulation`, `computePerVertNormals`, `boolean`,
`decimateMesh`). The only non-MeshLib additions are the `fromArray` / `toArray` bridges between a
container and a flat `Float32Array` / `Uint32Array`.

**Memory:** every value returned across the boundary is an embind handle you must `.delete()` —
the meshes and every container (`VertCoords` / `Triangulation`, and the results of `mesh.points`,
`mesh.topology`, `getTriangulation()`, `computePerVertNormals()`). Don't chain
`mesh.topology.getTriangulation()` without keeping the intermediate, or the `MeshTopology` handle
leaks.

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
