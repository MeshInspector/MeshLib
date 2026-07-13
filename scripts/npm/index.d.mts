/**
 * TypeScript surface for @meshinspector/meshlib (this file is the package's `types` entry).
 *
 * `bindings.d.mts` is generated from the embind registry by Emscripten's `--emit-tsd`.
 * Everything that crosses the JS boundary as a registered embind type is already precisely
 * typed there, and re-exported unchanged by the `export *` below.
 *
 * This overlay refines only the members that `--emit-tsd` emits as `any` because they cross
 * the boundary as a raw `emscripten::val`: the dynamically-shaped result objects, the
 * typed-array converters, the `val` parameters, and the progress-callback setter.
 *
 * Mechanism: an exported local declaration shadows the same-named `export *` re-export, so the
 * aliases below refine the generated ones. `MeshLibModule` `Omit`s the dynamic members from the
 * generated module type and re-adds them with precise signatures. Everything references
 * `Base.<name>`, so if a refined name ever drifts from the bindings the reference fails to
 * resolve and `tsc` errors here instead of shipping a wrong type.
 *
 * The specifier `./bindings.mjs` resolves (type-only) to `bindings.d.mts`; there is no runtime
 * `bindings.mjs` and none is needed — this is a declaration file, and the actual runtime exports
 * come from `meshlib.mjs` (the package `exports.import`).
 *
 * Refined instance types apply when a value is typed via the exported type — e.g.
 * `ml.VertCoords.fromArray(a)` or `const m: Mesh = ...`. A value reached only through an
 * un-refined return chain may still surface `any` at the deepest instance call; annotate with
 * the exported type to recover precision.
 */
import type * as Base from './bindings.mjs';

export * from './bindings.mjs';

// ---- Result shapes (built as emscripten::val objects by the bindings) --------------------

export interface PointOnFace {
  face: number;
  point: Base.Vector3f;
}

export interface MeshTriPoint {
  e: number;
  bary: { a: number; b: number };
}

export interface MeshProjectionResult {
  proj: PointOnFace;
  mtp: MeshTriPoint;
  distSq: number;
  valid: boolean;
}

export interface SignedMeshProjectionResult {
  proj: PointOnFace;
  mtp: MeshTriPoint;
  dist: number;
}

export interface MeshDistanceResult {
  a: PointOnFace;
  b: PointOnFace;
  distSq: number;
}

export interface CutMeshResult {
  resultCut: Uint32Array[];
  fbsWithContourIntersections: Base.FaceBitSet;
}

export interface ComponentsMapResult {
  map: Face2RegionMap;
  numRegions: number;
}

export interface LargeByAreaRegionsResult {
  faces: Base.FaceBitSet;
  numRegions: number;
}

export interface GridMinMax {
  min: number;
  max: number;
}

// The MeshComponents::FaceIncidence enum, taken from a sibling method's signature so the exact
// generated enum-type name never has to be guessed.
type FaceIncidence = Parameters<Base.MainModule['MeshComponents']['getComponent']>[2];

// ---- Refined instance types --------------------------------------------------------------
// A local `export` shadows the same-named `export *` re-export from the generated base.

type WithToArray<T, A> = Omit<T, 'toArray'> & { toArray(): A };
type WithToIndices<T> = Omit<T, 'toIndices'> & { toIndices(): Uint32Array };

export type VertCoords = WithToArray<Base.VertCoords, Float32Array>;
export type FaceNormals = WithToArray<Base.FaceNormals, Float32Array>;
export type Triangulation = WithToArray<Base.Triangulation, Uint32Array>;
export type VertMap = WithToArray<Base.VertMap, Uint32Array>;
export type FaceMap = WithToArray<Base.FaceMap, Uint32Array>;
export type Face2RegionMap = WithToArray<Base.Face2RegionMap, Uint32Array>;
export type VertColors = WithToArray<Base.VertColors, Uint8Array>;

export type BitSet = WithToIndices<Base.BitSet>;
export type FaceBitSet = WithToIndices<Base.FaceBitSet>;
export type VertBitSet = WithToIndices<Base.VertBitSet>;
export type EdgeBitSet = WithToIndices<Base.EdgeBitSet>;
export type UndirectedEdgeBitSet = WithToIndices<Base.UndirectedEdgeBitSet>;

export type MeshTopology = Omit<Base.MeshTopology, 'getTriangulation'> & {
  getTriangulation(): Triangulation;
};

export type Mesh = Omit<Base.Mesh, 'points' | 'topology' | 'toTriPoint'> & {
  readonly points: VertCoords;
  readonly topology: MeshTopology;
  toTriPoint( f: number, p: Base.Vector3f ): MeshTriPoint;
};

export type SelfIntersectionsSettings = Omit<Base.SelfIntersectionsSettings, 'callback'> & {
  // setter accepts a JS callback; embind emits the getter, so the base sees only `any`.
  callback: ( progress: number ) => boolean;
};

// ---- Refined module surface --------------------------------------------------------------

type WithFromArray<M, A, R> = Omit<M, 'fromArray'> & { fromArray( array: A ): R };
// The bitset module members carry a nullary constructor; `Omit` drops construct signatures, so
// re-add `new()` alongside the refined `fromIndices` to keep `new ml.FaceBitSet()` working.
type WithFromIndices<M, R> = Omit<M, 'fromIndices'> & {
  new(): R;
  fromIndices( indices: readonly number[] | Uint32Array ): R;
};

// Static `fromArray` converters (module-side); return the refined instance types above.
type Converters = {
  VertCoords: WithFromArray<Base.MainModule['VertCoords'], Float32Array, VertCoords>;
  FaceNormals: WithFromArray<Base.MainModule['FaceNormals'], Float32Array, FaceNormals>;
  Triangulation: WithFromArray<Base.MainModule['Triangulation'], Uint32Array, Triangulation>;
  VertMap: WithFromArray<Base.MainModule['VertMap'], Uint32Array, VertMap>;
  FaceMap: WithFromArray<Base.MainModule['FaceMap'], Uint32Array, FaceMap>;
  Face2RegionMap: WithFromArray<Base.MainModule['Face2RegionMap'], Uint32Array, Face2RegionMap>;
  VertColors: WithFromArray<Base.MainModule['VertColors'], Uint8Array, VertColors>;
  BitSet: WithFromIndices<Base.MainModule['BitSet'], BitSet>;
  FaceBitSet: WithFromIndices<Base.MainModule['FaceBitSet'], FaceBitSet>;
  VertBitSet: WithFromIndices<Base.MainModule['VertBitSet'], VertBitSet>;
  EdgeBitSet: WithFromIndices<Base.MainModule['EdgeBitSet'], EdgeBitSet>;
  UndirectedEdgeBitSet: WithFromIndices<Base.MainModule['UndirectedEdgeBitSet'], UndirectedEdgeBitSet>;
};

// Top-level free functions that return / accept a raw `val`.
type Functions = {
  findProjection( pt: Base.Vector3f, m: Base.Mesh ): MeshProjectionResult;
  findSignedDistance( pt: Base.Vector3f, m: Base.Mesh ): SignedMeshProjectionResult | null;
  findDistance( a: Base.Mesh, b: Base.Mesh ): MeshDistanceResult;
  cutMesh( mesh: Base.Mesh, contours: Base.OneMeshContours, params: Base.CutMeshParameters ): CutMeshResult;
  cutMeshByProjection( mesh: Base.Mesh, contours: readonly Float32Array[], settings: Base.CutByProjectionSettings ): Uint32Array[];
  findRightBoundary( topology: Base.MeshTopology ): Uint32Array[];
  fillHoles( mesh: Base.Mesh, edges: readonly number[] | Uint32Array, params: Base.FillHoleParams ): void;
  evalGridMinMax( grid: Base.FloatGrid ): GridMinMax;
};

// Tag-struct module-classes with `val`-returning static methods.
type ModuleClasses = {
  MeshComponents: Omit<Base.MainModule['MeshComponents'], 'getAllComponents' | 'getAllComponentsMap' | 'getLargeByAreaRegions'> & {
    getAllComponents( m: Base.Mesh, inc: FaceIncidence ): Base.FaceBitSet[];
    getAllComponentsMap( m: Base.Mesh, inc: FaceIncidence ): ComponentsMapResult;
    getLargeByAreaRegions( m: Base.Mesh, face2RegionMap: Base.Face2RegionMap, numRegions: number, minArea: number ): LargeByAreaRegionsResult;
  };
  VoxelsLoad: Omit<Base.MainModule['VoxelsLoad'], 'fromAnySupportedFormat'> & {
    fromAnySupportedFormat( path: string ): Base.VdbVolume[];
  };
};

// Deliberately left as generated (`any`): the `MultiwayICP` constructor's `val` argument — its
// module member is a construct signature, which `Omit` cannot target without dropping any other
// statics; refine as a follow-up if a precise `MeshOrPointsXf[]` input is wanted.

export type MeshLibModule =
  Omit<Base.MainModule, keyof Converters | keyof Functions | keyof ModuleClasses>
  & Converters
  & Functions
  & ModuleClasses;

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

declare function createMeshLib( options?: CreateMeshLibOptions ): Promise<MeshLibModule>;
export default createMeshLib;
