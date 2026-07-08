# @meshinspector/meshlib

[MeshLib](https://meshlib.io/) geometry library compiled to WebAssembly.

This package ships the **single-threaded** build of MeshLib's WebAssembly
module (`meshlib.mjs` + `meshlib.wasm`).

- Source: https://github.com/MeshInspector/MeshLib
- Documentation: https://meshlib.io/documentation/index.html

## Requirements

- Node.js >= 18

## Install

```sh
npm install @meshinspector/meshlib
```

## Use from CDN

In the browser you can skip npm entirely and import the module directly:

```js
// latest version
import createMeshLib from 'https://cdn.meshlib.io/js/meshlib/meshlib.mjs';

// or pin a specific version
import createMeshLib from 'https://cdn.meshlib.io/js/meshlib@1.2.3/meshlib.mjs';
```

## Usage

The default export is an async factory. Await it once to get the module instance, then
call MeshLib functions on it:

```js
import createMeshLib from '@meshinspector/meshlib';

const ml = await createMeshLib();

// Build a cube (side 2) from raw geometry.
const positions = new Float32Array([
  -1, -1, -1,   1, -1, -1,   1, 1, -1,   -1, 1, -1,
  -1, -1,  1,   1, -1,  1,   1, 1,  1,   -1, 1,  1,
]);
const indices = new Uint32Array([
  0, 2, 1,  0, 3, 2,  4, 5, 6,  4, 6, 7,  0, 1, 5,  0, 5, 4,
  3, 6, 2,  3, 7, 6,  0, 4, 7,  0, 7, 3,  1, 2, 6,  1, 6, 5,
]);

const coords = ml.VertCoords.fromArray(positions);
const tris = ml.Triangulation.fromArray(indices);
const mesh = ml.Mesh.fromTriangles(coords, tris);

console.log('volume =', mesh.volume()); // ~8

// Objects are backed by WebAssembly memory — free them explicitly.
coords.delete();
tris.delete();
mesh.delete();
```

## TypeScript

The package ships type definitions, so `createMeshLib` and the whole module API are typed with
no extra setup:

```ts
import createMeshLib, { type Mesh } from '@meshinspector/meshlib';

const ml = await createMeshLib();
const mesh: Mesh = ml.Mesh.fromTriangles(coords, tris);
const { valid, distSq } = ml.findProjection(point, mesh);
```

## Memory management

Values returned from the API (meshes, bit sets, settings, result objects, …) hold
WebAssembly memory that the JavaScript garbage collector does not reclaim. Call
`.delete()` on them when you are done to avoid leaks.

## License

Free for non-commercial and educational use. See [LICENSE](./LICENSE).

For commercial use, contact us at https://meshlib.io/book-a-call/
