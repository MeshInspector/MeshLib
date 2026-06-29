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

  const posAttr = geo.getAttribute( 'position' );
  const positions = posAttr.array instanceof Float32Array
    ? posAttr.array
    : Float32Array.from( posAttr.array );
  const idxArr = geo.index.array;
  const indices = idxArr instanceof Uint32Array
    ? idxArr
    : Uint32Array.from( idxArr );

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
  const points = mesh.points;
  const topology = mesh.topology;
  const tris = topology.getTriangulation();
  const exported = {
    positions: points.toArray(),
    indices: tris.toArray(),
  };
  points.delete();
  topology.delete();
  tris.delete();
  if ( normals ) {
    const vertNormals = ml.computePerVertNormals( mesh );
    exported.normals = vertNormals.toArray();
    vertNormals.delete();
  }
  return geometryToThree( THREE, exported );
}
