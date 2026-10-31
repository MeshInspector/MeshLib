# How to Install MeshLib SDK for JavaScript {#MeshLibJsSetupGuide}

# Installing MeshLib SDK for JavaScript

[TOC]

[MeshLib](https://meshlib.io/) is available for JavaScript as the geometry library compiled to WebAssembly, with the **complete MeshLib API** generated from the C++ headers (the same way as the Python, C#, Rust and Java bindings) as ES modules with TypeScript declarations.
It runs both in Node.js and in the browser, and is published to npm as two packages that share an identical API:
`@meshinspector/meshlib` (single-threaded) and `@meshinspector/meshlib-mt` (multi-threaded).
To see it running in the browser before you install anything, open [demo.meshlib.io](https://demo.meshlib.io/).

## Prerequisites
Before installing MeshLib SDK for JavaScript, ensure you have the following:

### Runtime
MeshLib runs in Node.js or in a modern browser.
 - **Node.js 22 or newer** (24 or newer for the `using` declarations the examples use).
 - In the browser, the current versions of Chrome, Firefox, Safari and Edge.

[Download Node.js](https://nodejs.org/en/download).

### Code Editor
You can use any JavaScript- or TypeScript-compatible editor, such as Visual Studio Code, WebStorm, or any IDE you prefer.

[Download Visual Studio Code](https://code.visualstudio.com/download)

## Choosing a Package: Single-Threaded or Multi-Threaded
MeshLib ships in two flavors with an identical API:

 - `@meshinspector/meshlib`: the single-threaded build. The simplest choice, and the right one for most applications.
 - `@meshinspector/meshlib-mt`: the multi-threaded build. It uses worker threads to parallelize geometry operations for higher throughput.

The examples below use `@meshinspector/meshlib`. To use the multi-threaded build instead, install `@meshinspector/meshlib-mt` and change the import specifier; in Node.js nothing else changes. In the browser the page must additionally be [cross-origin isolated](\ref JsMtCrossOriginIsolation).

## Installation
### Installation via npm
Install the package from npm and enable ES module support in your project:
```sh
npm install @meshinspector/meshlib
npm pkg set type=module
```

### Migrating from the 3.x Packages
The packages up to 3.1.x shipped a hand-written Embind module with a part of the API; the generated API is different:
 - The default export (`import createMeshLib`, `const ml = await createMeshLib()`) became a namespace import and an `init()` call: `import * as ml from "@meshinspector/meshlib"; await ml.init();`.
 - On the CDN the entry point is `index.mjs` instead of `meshlib.mjs`.
 - The classes, functions and fields follow the C++ API one to one (see [How the API Maps to C++](#MeshLibJsApiMapping) below), so the helpers of the old module have no direct counterparts: look up the C++ function instead.
 - Objects are still freed with `.delete()` (or `using`), and the typed-array conversions are `toArray()`/`view()`/`fromArray()`.

### Use from CDN
In the browser you can skip npm entirely and import the module directly:
```js
// latest release
import * as ml from 'https://js.meshlib.io/meshlib/index.mjs';

// or pin a release
import * as ml from 'https://js.meshlib.io/meshlib@v1.2.3.456/index.mjs';
```

The unpinned `index.mjs` re-exports the latest pinned release, which holds the modules themselves (`meshlib@v1.2.3.456/MeshLibC2.wasm` and the like).

#### Multi-Threaded Build
The multi-threaded build cannot run its workers straight from the CDN, because the browser refuses to run a worker script from another origin, so pass the fetched module to `init()` as a `Blob` and its worker pool starts from a same-origin `blob:` URL. The module must be the one of the imported release, so pin it:
```js
import * as ml from 'https://js.meshlib.io/meshlib-mt@v1.2.3.456/index.mjs';

const url = 'https://js.meshlib.io/meshlib-mt@v1.2.3.456/MeshLibC2.mjs';
const blob = new Blob( [ await ( await fetch( url ) ).text() ], { type: 'text/javascript' } );
await ml.init( { mainScriptUrlOrBlob: blob } );
```

In the browser the multi-threaded build additionally requires the page to be [cross-origin isolated](\ref JsMtCrossOriginIsolation).

### Bundlers
The package is a set of ES modules plus the wasm sidecar `MeshLibC2.wasm`. Vite and webpack 5 resolve the wasm from the module and emit it as an asset, so a plain `import` needs no configuration. For other bundlers, such as esbuild or Rollup, import the wasm as an asset URL and hand it to the loader via `locateFile`:
```js
import * as ml from '@meshinspector/meshlib';
import wasmUrl from '@meshinspector/meshlib/MeshLibC2.wasm';

await ml.init( { locateFile: () => wasmUrl } );
```

The bundler must treat `.wasm` files as static assets, so that the import resolves to the URL of the emitted file; the option is usually called an asset or file loader. For example:

- **esbuild**: pass `--loader:.wasm=file`
- **Rollup**: add `@rollup/plugin-url` with `include: /\.wasm$/`
- **Parcel**: use the `url:` scheme on the import specifier: `import wasmUrl from "url:@meshinspector/meshlib/MeshLibC2.wasm";`

The options of `init()` go to the Emscripten module factory (`locateFile`, `print`, `printErr`, `mainScriptUrlOrBlob`, ...).

### Payload Size {#MeshLibJsPayloadSize}
MeshLib is a WebAssembly build of the full geometry library with its complete API exported, so a browser downloads several tens of megabytes of wasm (a fraction of that over the wire: the CDN serves it Brotli-compressed) before the first geometry call. `@meshinspector/meshlib-mt` is the same order of magnitude.

Consequences for a browser application:

 - The `.wasm` is a sidecar file fetched as a **second request**, separate from the `.mjs` modules. Serve it as `application/wasm` with compression enabled, or the download is several times larger than it needs to be.
 - The JavaScript side is many small modules (one per C++ header); a bundler merges them, and since the package declares `"sideEffects": false` it can drop the ones you don't use.
 - `npm install` unpacks the browser and the Node.js wasm side by side. Only one of them ever reaches the browser; the rest is bundler input, not payload.

If the geometry is not needed on first paint, load the module lazily instead of at the top level, so the wasm download does not block startup:
```js
// load MeshLib only when the user actually needs geometry
const ml = await import('@meshinspector/meshlib');
await ml.init();
```

## Cross-Origin Isolation for the Multi-Threaded Build in the Browser {#JsMtCrossOriginIsolation}
This section applies to browsers only. In Node.js the multi-threaded build needs no headers, flags, or extra configuration.

The multi-threaded package `@meshinspector/meshlib-mt` relies on `SharedArrayBuffer`, which browsers only enable on **cross-origin isolated** pages. The server that serves the page loading the module must send these headers:
```unparsed
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Without them `crossOriginIsolated` is `false`, `SharedArrayBuffer` is unavailable, and `init()` never resolves.

For a Vite dev server, set the headers in `vite.config.js` (whatever hosts the production build must send them too; `vite-plugin-cross-origin-isolation` can stamp them for `vite preview`):
```js
// vite.config.js
export default {
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
};
```

Where you cannot set response headers at all — GitHub Pages and similar static hosting — the usual workaround is a `coi-serviceworker`-style shim: a service worker that re-serves the page with the two headers and reloads it once. MeshLib's own [interactive demo](https://demo.meshlib.io/) is hosted exactly this way.

So on a page that is not cross-origin isolated you have three options: send the headers, install the service-worker shim, or use the single-threaded `@meshinspector/meshlib` package, which has no such requirement.

## Getting Started: Your First Example
Import the package as a namespace and call `init()` once: it downloads and instantiates the wasm, after which the whole MeshLib API is there, the classes and functions of the `MR` namespace at the top level and the nested namespaces (`MeshLoad`, `MeshSave`, `MeshBuilder`, ...) as namespace objects:

> [!NOTE]
> The `using` declaration requires Node.js 24+ or a current browser. On older runtimes, call `.delete()` on each object instead (see the **Memory Management** section below).

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
```

> [!NOTE]
> This example loads the module at the top level, which downloads the wasm on startup. In the browser, consider the lazy alternative in [Payload Size](\ref MeshLibJsPayloadSize).

## How the API Maps to C++ {#MeshLibJsApiMapping}
The bindings follow the C++ API: the same classes, functions, parameters and default values, documented in the [C++ reference](\ref APIPage). The differences come from JavaScript itself:

 - **Fields are properties**: `settings.maxError = 0.05`, `mesh.topology`, `box.min.x`. The math types (`Vector3f`, `Matrix3f`, `AffineXf3f`, `Box3f`, the ids) are small structs with accessor properties over wasm memory.
 - **Overloads** keep their names and are resolved by the argument types at runtime. The few overloads that JavaScript can't tell apart (`int` vs `float` parameters) are exported under the names of their C bindings as well, e.g. `foo_float`.
 - **Default arguments**: trailing parameters with defaults can be omitted, any parameter with a default takes `null` to use it.
 - **Return values**: `std::string` results are strings, `Expected` results are unwrapped and an error throws `UnexpectedResultException` with the message, C++ exceptions throw `CppException`. Functions returning references give you references that keep their owner alive.
 - **Callbacks**: `std::function` parameters (progress callbacks and such) take a plain JavaScript function.
 - **Containers**: `VertCoords`, `Triangulation`, `std::vector`s and the other contiguous containers have `toArray()` (a typed array copy), `view()` (a typed array over the wasm memory, valid until the next allocation), `setFromArray()` and `static fromArray()`.
 - **Mesh parts**: the functions taking `MeshPart` in C++ take a `new ml.MeshPart(mesh)` (optionally with a face region).
 - **Enums** are plain numbers with named constants: `ml.BooleanOperation.Union`.

## TypeScript
The package ships type definitions for the whole API, with the doc comments of the C++ headers:
```ts
import * as ml from '@meshinspector/meshlib';

await ml.init();
const mesh: ml.Mesh = ml.makeUVSphere(1, 32, 32);
const settings = new ml.DecimateSettings();
settings.maxError = 0.05;
const result: ml.DecimateResult = ml.decimateMesh(mesh, settings);
mesh.delete();
```

## Memory Management {#JsMemoryManagement}
Values returned from the API (meshes, bit sets, settings, result objects, and so on) hold WebAssembly memory. They are released when the JavaScript garbage collector collects them, but it does so late and unpredictably, so free the large ones explicitly.

The preferred way is JavaScript's explicit resource management: declare a handle with [`using`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/using) and it is freed automatically when its scope ends, even if an exception is thrown.
```js
using mesh = ml.Mesh.fromTriangles(coords, tris);
// ... use mesh; it is freed at the end of this scope
```

When the number of handles is dynamic (for example built in a loop), collect them in a [`DisposableStack`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/DisposableStack), which frees everything it holds, in reverse order, at the end of the scope:
```js
using stack = new DisposableStack();
for (const path of inputPaths) {
  const cloud = stack.use(ml.PointsLoad.fromAnySupportedFormat(path));
  // ... use cloud
}
// every handle passed to stack.use(...) is freed here
```

`using` and `DisposableStack` are part of JavaScript's Explicit Resource Management, available in Node.js 24+ and current browsers. On older runtimes and browsers, call `.delete()` on each object when you are done instead:
```js
const mesh = ml.Mesh.fromTriangles(coords, tris);
// ... use mesh
mesh.delete();
```

References into other objects (`mesh.points`, `mesh.topology`, the elements returned by `index()`) own nothing and need no freeing; they keep their owner alive while they exist.

## Files in the Browser
In Node.js, `init()` picks the build of the module with access to the real file system, so `MeshLoad`, `MeshSave` and the other file functions take paths. In the browser the module has an in-memory file system (`ml.ffi.M.FS`): write the bytes you fetched into it, then load from that path, and read the saved files back from it.

## Try Interactive MeshLib Examples
You can check how MeshLib works in a browser environment with live interactive examples: https://demo.meshlib.io/

## Try MeshLib with JavaScript Examples
After installing MeshLib, a great way to start exploring its capabilities is through the code samples. Each of the following example pages includes a JavaScript tab:

 - [**Loading and saving mesh files**](\ref ExampleMeshLoadSave)
 - [**Mesh Boolean**](\ref ExampleMeshBoolean)
 - [**Mesh Decimation**](\ref ExampleMeshDecimate)

Browse the full set on the [**MeshLib Code Samples**](\ref Examples) page.
