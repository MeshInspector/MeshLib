import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import createMeshLib from './meshlib.mjs';
import { meshFromThree, meshToThree } from './meshlib-three.js';

const statsEl = document.getElementById( 'stats' );
const errEl = document.getElementById( 'err' );

const app = document.getElementById( 'app' );
const renderer = new THREE.WebGLRenderer( { antialias: true } );
renderer.setPixelRatio( devicePixelRatio );
renderer.setSize( innerWidth, innerHeight );
app.appendChild( renderer.domElement );

const scene = new THREE.Scene();
scene.background = new THREE.Color( 0x1a1c20 );

const camera = new THREE.PerspectiveCamera( 45, innerWidth / innerHeight, 0.1, 100 );
camera.position.set( 3.2, 2.4, 3.6 );

const controls = new OrbitControls( camera, renderer.domElement );
controls.enableDamping = true;

scene.add( new THREE.HemisphereLight( 0xffffff, 0x39404d, 1.15 ) );
const key = new THREE.DirectionalLight( 0xffffff, 1.4 );
key.position.set( 4, 6, 3 );
scene.add( key );

const material = new THREE.MeshStandardMaterial( { color: 0x5b9bd5, metalness: 0.1, roughness: 0.55 } );
let resultMesh = null;

addEventListener( 'resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize( innerWidth, innerHeight );
} );

( function loop() {
  requestAnimationFrame( loop );
  controls.update();
  renderer.render( scene, camera );
} )();

const ml = await createMeshLib();

const boxGeom = new THREE.BoxGeometry( 1.6, 1.6, 1.6, 6, 6, 6 );
const sphereGeom = new THREE.SphereGeometry( 1.05, 48, 32 );
sphereGeom.translate( 0.7, 0.7, 0.7 );

function recompute() {
  errEl.textContent = '';
  const op = document.getElementById( 'op' ).value;
  const ratio = Number( document.getElementById( 'ratio' ).value ) / 100;

  const a = meshFromThree( ml, THREE, boxGeom, { mergeVertices } );
  const b = meshFromThree( ml, THREE, sphereGeom, { mergeVertices } );
  const res = ml.boolean( a, b, ml.BooleanOperation[op] );
  if ( !res.valid() ) {
    errEl.textContent = 'boolean failed: ' + res.errorString;
    a.delete(); b.delete(); res.delete();
    return;
  }
  const out = res.mesh;

  out.pack();
  const outTopology = out.topology;
  const outTris = outTopology.getTriangulation();
  const trisAfterBool = outTris.toArray().length / 3;
  outTopology.delete();
  outTris.delete();
  let decimated = { facesDeleted: 0 };
  if ( ratio < 1 ) {
    const s = new ml.DecimateSettings();
    s.maxDeletedFaces = Math.floor( trisAfterBool * ( 1 - ratio ) );
    decimated = ml.decimateMesh( out, s );
    s.delete();
  }

  const geom = meshToThree( ml, THREE, out, { normals: true } );
  a.delete(); b.delete(); res.delete(); out.delete();

  if ( resultMesh ) {
    scene.remove( resultMesh );
    resultMesh.geometry.dispose();
  }
  resultMesh = new THREE.Mesh( geom, material );
  scene.add( resultMesh );

  const finalTris = geom.index.count / 3;
  statsEl.textContent =
    `boolean result: ${trisAfterBool.toLocaleString()} triangles\n` +
    `after decimate: ${finalTris.toLocaleString()} triangles ` +
    `(removed ${decimated.facesDeleted.toLocaleString()})`;
}

document.getElementById( 'ratio' ).addEventListener( 'input', ( e ) => {
  document.getElementById( 'ratioVal' ).textContent = e.target.value;
} );
document.getElementById( 'run' ).addEventListener( 'click', recompute );

recompute();
