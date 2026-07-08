# @meshinspector/meshlib-mt

[MeshLib](https://meshlib.io/) geometry library compiled to WebAssembly — the **multi-threaded** build.

This is the higher-throughput sibling of [`@meshinspector/meshlib`](https://www.npmjs.com/package/@meshinspector/meshlib)
(single-threaded). It uses worker threads and `SharedArrayBuffer` to parallelize geometry operations.

- Source: https://github.com/MeshInspector/MeshLib
- Documentation: https://meshlib.io/documentation/index.html

## Requirements

- Node.js >= 21

## Install

```sh
npm install @meshinspector/meshlib-mt
```

## Use from CDN

In the browser you can skip npm entirely and import the module directly:

```js
// latest version
import createMeshLib from 'https://cdn.meshlib.io/js/meshlib-mt/meshlib-mt.mjs';

// or pin a specific version
import createMeshLib from 'https://cdn.meshlib.io/js/meshlib-mt@1.2.3/meshlib-mt.mjs';
```

## Browser requirements: cross-origin isolation

The multi-threaded build relies on `SharedArrayBuffer`, which browsers only enable on
**cross-origin isolated** pages. Your server must send these headers with the page that
loads the module:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Without them `SharedArrayBuffer` is unavailable and the module will fail to initialize —
use the single-threaded [`@meshinspector/meshlib`](https://www.npmjs.com/package/@meshinspector/meshlib)
package in that case.

## Usage

The default export is an async factory. Await it once to get the module instance, then call
MeshLib functions on it:

```js
import createMeshLib from '@meshinspector/meshlib-mt';

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
import createMeshLib, { type Mesh } from '@meshinspector/meshlib-mt';

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
