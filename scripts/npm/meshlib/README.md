# @meshinspector/meshlib

[MeshLib](https://meshlib.io/) geometry library compiled to WebAssembly, with the **complete MeshLib API**
generated from the C++ headers (the same way as the Python, C#, Rust and Java bindings), as ES modules with
TypeScript declarations.

This package ships the **single-threaded** build. The multi-threaded build is
[`@meshinspector/meshlib-mt`](https://www.npmjs.com/package/@meshinspector/meshlib-mt), with an identical API.

- Source: https://github.com/MeshInspector/MeshLib
- Documentation: https://meshlib.io/documentation/index.html

## Requirements

- Node.js >= 22 (>= 24 for `using`), or a current browser.

## Install

```sh
npm install @meshinspector/meshlib
```

## Usage

Import the package as a namespace, initialize it once (this downloads and instantiates the wasm), then the
whole MeshLib API is there: the classes of the `MR` namespace at the top level, the nested namespaces
(`MeshLoad`, `MeshSave`, `MeshBuilder`, ...) as namespace objects.

> `using` requires Node.js 24+ or a current browser. On older runtimes, call `.delete()`
> instead — see [Memory management](#memory-management).

```js
import * as ml from '@meshinspector/meshlib';

await ml.init();

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

// Any MeshLib function: a boolean with a sphere, then the result back as typed arrays.
using sphere = ml.makeUVSphere(1.5, 32, 32);
using result = ml.boolean(mesh, sphere, ml.BooleanOperation.Intersection);
console.log('intersection volume =', result.mesh.volume());
const outPositions = result.mesh.points.toArray();                  // Float32Array, x y z per vertex
const outIndices = result.mesh.topology.getTriangulation().toArray(); // Int32Array, 3 vertex ids per triangle
```

The API follows the C++ one: see the [C++ reference](https://meshlib.io/documentation/index.html), the
same classes, functions, parameters and defaults. What's different:

- Fields are properties (`settings.maxError = 0.05`, `mesh.topology`), overloads are resolved by the
  argument types, `std::string` results are strings, `Expected` results are unwrapped (an error throws
  `UnexpectedResultException`), C++ exceptions throw `CppException`.
- Parameters with defaults can be omitted or passed as `null`; `std::function` parameters (progress
  callbacks) take a plain JS function.
- The contiguous containers (`VertCoords`, `Triangulation`, `std::vector`s, ...) have `toArray()`,
  `view()`, `setFromArray()` and `fromArray()` for typed arrays.
- Every `.d.mts` carries the doc comments of the C++ API, so editors show them.

## Memory management

Values returned from the API (meshes, bit sets, settings, result objects, ...) hold WebAssembly memory.
They are freed when the JavaScript garbage collector collects them, but it does so late and unpredictably,
so free the large ones explicitly: declare them with `using`, or call `.delete()`.

```js
using mesh = ml.makeUVSphere(1, 64, 64);
// ... use mesh; it is freed at the end of this scope
```

```js
const mesh = ml.makeUVSphere(1, 64, 64);
// ... use mesh
mesh.delete();
```

References into other objects (`mesh.points`, `mesh.topology`) don't own anything and keep their owner alive.

## Node.js and the browser

`init()` picks the Node build of the module (with access to the real file system, so `MeshLoad` and
`MeshSave` work with paths) under Node.js, and the browser build everywhere else. In the browser, file
functions see the module's in-memory file system (`ml.ffi.M.FS`); load files with `fetch` and pass the
bytes through it.

## Use from CDN

In the browser you can skip npm entirely and import the module directly:

```js
// latest release
import * as ml from 'https://js.meshlib.io/meshlib/index.mjs';

// or pin a release — v1.2.3.456 is an example tag, not a real one
import * as ml from 'https://js.meshlib.io/meshlib@v1.2.3.456/index.mjs';
```

To pin, take a real tag from the [releases page](https://github.com/MeshInspector/MeshLib/releases): it has a
leading `v` and four components, and is **not** the npm version — the tag `v1.2.3.456` would be npm `1.2.3-456`.

## Using with bundlers

The package is many small ES modules plus the wasm sidecar. Vite and webpack 5 resolve the `.wasm` from
the module and emit it as an asset, so a plain `import` needs no configuration. For other bundlers, import
the wasm as an asset URL and hand it to the loader via `locateFile`:

```js
import * as ml from '@meshinspector/meshlib';
import wasmUrl from '@meshinspector/meshlib/MeshLibC2.wasm';

await ml.init( { locateFile: () => wasmUrl } );
```

The options of `init()` go to the Emscripten module factory (`locateFile`, `print`, `printErr`, ...).

## TypeScript

The package ships type definitions for the whole API:

```ts
import * as ml from '@meshinspector/meshlib';

await ml.init();
const mesh: ml.Mesh = ml.makeUVSphere(1, 32, 32);
const settings = new ml.DecimateSettings();
settings.maxError = 0.05;
const result: ml.DecimateResult = ml.decimateMesh(mesh, settings);
mesh.delete();
```

Also make sure to add `"type": "module"` to your package.json.
