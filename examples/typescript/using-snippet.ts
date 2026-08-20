// The TypeScript snippet from the JavaScript Setup Guide, kept here so CI type-checks it.
// Compiled by `npm run build` (see tsconfig.json), but not executed: `using` needs Node.js 24+
// at run time, while the snippet only has to type-check.
import createMeshLib, { type Mesh } from '@meshinspector/meshlib';

const ml = await createMeshLib();

using coords = ml.VertCoords.fromArray(new Float32Array([
  -1, -1, -1,   1, -1, -1,   1, 1, -1,   -1, 1, -1,
  -1, -1,  1,   1, -1,  1,   1, 1,  1,   -1, 1,  1,
]));
using tris = ml.Triangulation.fromArray(new Uint32Array([
  0, 2, 1,  0, 3, 2,  4, 5, 6,  4, 6, 7,  0, 1, 5,  0, 5, 4,
  3, 6, 2,  3, 7, 6,  0, 4, 7,  0, 7, 3,  1, 2, 6,  1, 6, 5,
]));

// fromTriangles is declared as `Mesh | null`, so narrow it before use.
const built = ml.Mesh.fromTriangles(coords, tris);
if (!built) throw new Error('failed to build mesh');
using mesh: Mesh = built;

const { valid, distSq } = ml.findProjection({ x: 5, y: 0, z: 0 }, mesh);
console.log(valid, Math.sqrt(distSq)); // true 4
