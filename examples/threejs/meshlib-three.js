// Converts between Three.js `BufferGeometry` and MeshLib meshes through typed arrays.

export function meshFromThree( ml, THREE, geometry, { mergeVertices } = {} ) {
  let geo = geometry.clone();
  for ( const name of Object.keys( geo.attributes ) )
    if ( name !== 'position' ) geo.deleteAttribute( name );

  if ( mergeVertices ) {
    geo = mergeVertices( geo );
  } else if ( !geo.index ) {
    const n = geo.getAttribute( 'position' ).count;
    const trivial = new Uint32Array( n );
    for ( let i = 0; i < n; i++ ) trivial[i] = i;
    geo.setIndex( new THREE.BufferAttribute( trivial, 1 ) );
  }

  const positions = geo.getAttribute( 'position' ).array;
  const indices = geo.index.array;

  // `fromArray()` copies the scalars into the MeshLib containers: x, y, z per vertex, three vertex ids per triangle.
  const coords = ml.VertCoords.fromArray( positions );
  const tris = ml.Triangulation.fromArray( indices );
  const mesh = ml.Mesh.fromTriangles( coords, tris );
  coords.delete();
  tris.delete();
  return mesh;
}

export function geometryToThree( THREE, exported ) {
  const g = new THREE.BufferGeometry();
  g.setAttribute( 'position', new THREE.BufferAttribute( exported.positions, 3 ) );
  if ( exported.normals )
    g.setAttribute( 'normal', new THREE.BufferAttribute( exported.normals, 3 ) );
  g.setIndex( new THREE.BufferAttribute( exported.indices, 1 ) );
  if ( !exported.normals )
    g.computeVertexNormals();
  return g;
}

export function meshToThree( ml, THREE, mesh, { normals = false } = {} ) {
  mesh.pack(); // getTriangulation() is undefined on deleted faces; pack removes them
  // `toArray()` returns copies (Float32Array of the coordinates, Int32Array of the vertex ids), safe to keep.
  const tris = mesh.topology.getTriangulation();
  const ids = tris.toArray();
  tris.delete();
  const exported = {
    positions: mesh.points.toArray(),
    // three.js draws an Int32Array index as gl.INT, which drawElements rejects: reinterpret the ids as unsigned, without a copy.
    indices: new Uint32Array( ids.buffer, ids.byteOffset, ids.length ),
  };
  if ( normals ) {
    const vertNormals = ml.computePerVertNormals( mesh );
    exported.normals = vertNormals.toArray();
    vertNormals.delete();
  }
  return geometryToThree( THREE, exported );
}
