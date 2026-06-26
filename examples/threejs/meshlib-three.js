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

  return ml.Wasm.meshFromGeometry( positions, indices );
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
  const exported = ml.Wasm.meshToGeometry( mesh, normals );
  return geometryToThree( THREE, exported );
}
