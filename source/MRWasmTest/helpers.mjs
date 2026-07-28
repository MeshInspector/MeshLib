import { pathToFileURL } from 'node:url';

const moduleUrl = process.env.MESHLIB_MODULE
  ? pathToFileURL( process.env.MESHLIB_MODULE ).href
  : new URL( './meshlib.mjs', import.meta.url ).href;

const { default: createMeshLib } = await import( moduleUrl );
export const ml = await createMeshLib();

export function meshFromGeometry( positions, indices ) {
  const coords = ml.VertCoords.fromArray( positions );
  const tris = ml.Triangulation.fromArray( indices );
  const m = ml.Mesh.fromTriangles( coords, tris );
  coords.delete();
  tris.delete();
  return m;
}

export function meshToGeometry( mesh, wantNormals ) {
  mesh.pack();
  const points = mesh.points;
  const topology = mesh.topology;
  const tris = topology.getTriangulation();
  const out = { positions: points.toArray(), indices: tris.toArray() };
  points.delete();
  topology.delete();
  tris.delete();
  if ( wantNormals ) {
    const vertNormals = ml.computePerVertNormals( mesh );
    out.normals = vertNormals.toArray();
    vertNormals.delete();
  }
  return out;
}

export function cube( cx, cy, cz, s = 2 ) {
  const h = s / 2;
  const positions = new Float32Array( [
    cx - h, cy - h, cz - h,
    cx + h, cy - h, cz - h,
    cx + h, cy + h, cz - h,
    cx - h, cy + h, cz - h,
    cx - h, cy - h, cz + h,
    cx + h, cy - h, cz + h,
    cx + h, cy + h, cz + h,
    cx - h, cy + h, cz + h,
  ] );
  const indices = new Uint32Array( [
    0, 2, 1,  0, 3, 2,
    4, 5, 6,  4, 6, 7,
    0, 1, 5,  0, 5, 4,
    3, 6, 2,  3, 7, 6,
    0, 4, 7,  0, 7, 3,
    1, 2, 6,  1, 6, 5,
  ] );
  return { positions, indices };
}
