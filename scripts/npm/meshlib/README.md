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
import createMeshLib from 'https://js.meshlib.io/meshlib/meshlib.mjs';

// or pin a specific version
import createMeshLib from 'https://js.meshlib.io/meshlib@1.2.3/meshlib.mjs';
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

using coords = ml.VertCoords.fromArray(positions);
using tris = ml.Triangulation.fromArray(indices);
using mesh = ml.Mesh.fromTriangles(coords, tris);

console.log('volume =', mesh.volume()); // ~8
// `using` frees these WebAssembly-backed objects automatically at the end of scope
```

> `using` requires Node.js 24+ or a current browser. On older runtimes, call `.delete()`
> instead — see [Memory management](#memory-management).

## Using with bundlers

Bundlers (Vite, webpack, Rollup) hash and relocate the sidecar `meshlib.wasm`, so the module can't
locate it on its own. Import the wasm as an asset URL and hand it to the loader via `locateFile`:

```js
import createMeshLib from '@meshinspector/meshlib';
import wasmUrl from '@meshinspector/meshlib/meshlib.wasm?url';

const ml = await createMeshLib( { locateFile: () => wasmUrl } );
```

## TypeScript

The package ships type definitions, so `createMeshLib` and the whole module API are typed with
no extra setup:

```ts
import createMeshLib, { type Mesh } from '@meshinspector/meshlib';

const ml = await createMeshLib();
using mesh: Mesh = ml.Mesh.fromTriangles(coords, tris);
const { valid, distSq } = ml.findProjection(point, mesh);
```

## Memory management

Values returned from the API (meshes, bit sets, settings, result objects, …) hold
WebAssembly memory that the JavaScript garbage collector does not reclaim, so each one
must be freed explicitly.

The preferred way is JavaScript's explicit resource management: declare a handle with
`using` and it is freed automatically when its scope ends — even if an exception is thrown.

```js
using mesh = ml.Mesh.fromTriangles(coords, tris);
// ... use mesh; it is freed at the end of this scope
```

When the number of handles is dynamic (for example built in a loop), collect them in a
`DisposableStack`, which frees everything it holds, in reverse order, at the end of the scope:

```js
using stack = new DisposableStack();
for (const path of inputPaths) {
  const cloud = stack.use(ml.PointsLoad.fromAnySupportedFormat(path));
  // ... use cloud
}
// every handle passed to stack.use(...) is freed here
```

`using` and `DisposableStack` are part of JavaScript's Explicit Resource Management,
available in Node.js 24+ and current browsers. On older runtimes and browsers, call
`.delete()` on each object when you are done instead:

```js
const mesh = ml.Mesh.fromTriangles(coords, tris);
// ... use mesh
mesh.delete();
```

See also, on MDN: [`using`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/using) and [`DisposableStack`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/DisposableStack).

## License

Free for non-commercial and educational use. See [LICENSE](./LICENSE).

For commercial use, contact us at https://meshlib.io/book-a-call/
