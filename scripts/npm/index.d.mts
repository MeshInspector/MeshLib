/**
 * TypeScript surface for @meshinspector/meshlib (this file is the package's `types` entry).
 *
 * Everything that crosses the JS boundary is precisely typed in `bindings.d.mts`, generated from the
 * embind registry by Emscripten's `--emit-tsd`, and re-exported unchanged by the `export *` below.
 *
 * This file adds only what `--emit-tsd` cannot generate: the default-export factory under its real
 * name with typed options.
 *
 * The specifier `./bindings.mjs` resolves (type-only) to `bindings.d.mts`; there is no runtime
 * `bindings.mjs` and none is needed — the actual runtime exports come from `meshlib.mjs` (the package
 * `exports.import`).
 */
import type { MainModule } from './bindings.mjs';

export * from './bindings.mjs';

/** Options for the module factory (the Emscripten `Module` object). All optional. */
export interface CreateMeshLibOptions {
  /**
   * Resolve the URL of a runtime file — most importantly the sidecar `.wasm`. Bundlers should
   * pass this pointing at a `?url` import of the wasm; see "Using with Vite / bundlers" in the
   * README.
   */
  locateFile?: ( path: string, scriptDirectory: string ) => string;
  /** Multi-threaded build only: URL or Blob of the main script, so pthread workers can load it. */
  mainScriptUrlOrBlob?: string | Blob;
  print?: ( text: string ) => void;
  printErr?: ( text: string ) => void;
}

declare function createMeshLib( options?: CreateMeshLibOptions ): Promise<MainModule>;
export default createMeshLib;
