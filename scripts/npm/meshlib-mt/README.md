# @meshinspector/meshlib-mt

[MeshLib](https://meshlib.io/) geometry library compiled to WebAssembly — the **multi-threaded** build, with the
**complete MeshLib API** generated from the C++ headers as ES modules with TypeScript declarations.

This is the higher-throughput sibling of [`@meshinspector/meshlib`](https://www.npmjs.com/package/@meshinspector/meshlib)
(single-threaded), with an identical API. It uses worker threads and `SharedArrayBuffer` to parallelize
geometry operations.

- Source: https://github.com/MeshInspector/MeshLib
- Documentation: https://meshlib.io/documentation/index.html

## Requirements

- Node.js >= 22 (>= 24 for `using`), or a current browser on a [cross-origin isolated](#browser-cross-origin-isolation) page.

## Install

```sh
npm install @meshinspector/meshlib-mt
```

## Usage

The same as `@meshinspector/meshlib`, see its README for the API overview, memory management, bundlers
and TypeScript. In short:

```js
import * as ml from '@meshinspector/meshlib-mt';

await ml.init();

using mesh = ml.makeUVSphere(1, 64, 64);
using settings = new ml.DecimateSettings();
settings.maxError = 0.01;
settings.subdivideParts = 16; // decimated in parallel
ml.decimateMesh(mesh, settings);
console.log('faces after decimation =', mesh.topology.numValidFaces());
```

## Browser: cross-origin isolation

The multi-threaded build relies on `SharedArrayBuffer`, which browsers only enable on cross-origin isolated
pages. The server must send these headers for the page that loads the module:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Without them `crossOriginIsolated` is `false` and `init()` never resolves. Where you cannot set response
headers (GitHub Pages and similar static hosting), a `coi-serviceworker`-style shim re-serves the page with
the headers. In Node.js nothing is needed.

## Use from CDN

The browser refuses to run a worker script from another origin, so the worker pool of the module must start
from a same-origin URL: fetch the module and pass it to `init()` as a `Blob`:

```js
import * as ml from 'https://js.meshlib.io/meshlib-mt/index.mjs';

const url = 'https://js.meshlib.io/meshlib-mt/MeshLibC2.mjs';
const blob = new Blob( [ await ( await fetch( url ) ).text() ], { type: 'text/javascript' } );
await ml.init( { mainScriptUrlOrBlob: blob } );
```

Pin a release by its tag in the URL (`https://js.meshlib.io/meshlib-mt@v1.2.3.456/index.mjs`), see the
[releases page](https://github.com/MeshInspector/MeshLib/releases).
