import assert from 'node:assert/strict';
import { ml, cube, meshFromGeometry } from './helpers.mjs';

{
  const c = cube( 0, 0, 0, 2 );
  const m = meshFromGeometry( c.positions, c.indices );

  const s = new ml.SelfIntersectionsSettings();
  let calls = 0;
  s.callback = ( progress ) => { calls++; return true; };
  ml.SelfIntersections.fix( m, s );
  assert.ok( calls >= 1, 'fix invoked the JS progress callback' );

  const s2 = new ml.SelfIntersectionsSettings();
  s2.callback = () => false;
  assert.throws( () => ml.SelfIntersections.fix( m, s2 ), /cancel/i,
    'a cancelled fix surfaces the Expected<void> error as a JS exception' );

  s.delete(); s2.delete(); m.delete();
}

{
  const c = cube( 0, 0, 0, 2 );
  const m = meshFromGeometry( c.positions, c.indices );

  const faces = ml.SelfIntersections.getFaces( m, true );
  assert.ok( faces instanceof ml.FaceBitSet, 'getFaces returns a FaceBitSet' );
  assert.equal( faces.count(), 0, 'a clean cube has no self-intersecting faces' );

  faces.delete(); m.delete();
}
