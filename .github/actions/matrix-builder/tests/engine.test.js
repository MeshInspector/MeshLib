'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const {
  buildMatrix,
  cartesian,
  isTruthy,
  matchesAll,
} = require('../lib/engine.js');

// Compare two arrays of objects irrespective of key order within each object.
// GitHub consumes the matrix output via fromJSON(), so field order is
// semantically irrelevant; we test the structure, not the byte layout.
function canon(arr) {
  return arr.map(o => {
    const out = {};
    for (const k of Object.keys(o).sort()) out[k] = o[k];
    return out;
  });
}

function deepEqual(a, b, msg) {
  if (msg === undefined) assert.deepStrictEqual(canon(a), canon(b));
  else assert.deepStrictEqual(canon(a), canon(b), msg);
}

test('cartesian: empty axes', () => {
  assert.deepStrictEqual(cartesian({}), []);
});

test('cartesian: single axis', () => {
  assert.deepStrictEqual(
    cartesian({ a: [1, 2, 3] }),
    [{ a: 1 }, { a: 2 }, { a: 3 }],
  );
});

test('cartesian: two axes, lexicographic order', () => {
  assert.deepStrictEqual(
    cartesian({ a: [1, 2], b: ['x', 'y'] }),
    [
      { a: 1, b: 'x' },
      { a: 1, b: 'y' },
      { a: 2, b: 'x' },
      { a: 2, b: 'y' },
    ],
  );
});

test('cartesian: scalar axis value treated as one-element list', () => {
  assert.deepStrictEqual(
    cartesian({ a: 'only', b: [1, 2] }),
    [{ a: 'only', b: 1 }, { a: 'only', b: 2 }],
  );
});

test('isTruthy: literals', () => {
  assert.equal(isTruthy(true), true);
  assert.equal(isTruthy(false), false);
  assert.equal(isTruthy(null), false);
  assert.equal(isTruthy(undefined), false);
});

test('isTruthy: strings', () => {
  for (const s of ['', 'false', 'False', 'FALSE', '0', 'no', 'No', 'off', '  false  ']) {
    assert.equal(isTruthy(s), false, `'${s}' should be falsy`);
  }
  for (const s of ['true', 'True', '1', 'yes', 'on', 'anything', ' x ']) {
    assert.equal(isTruthy(s), true, `'${s}' should be truthy`);
  }
});

test('matchesAll: subset semantics', () => {
  assert.equal(matchesAll({ a: 1, b: 2 }, { a: 1 }), true);
  assert.equal(matchesAll({ a: 1, b: 2 }, { a: 1, b: 2 }), true);
  assert.equal(matchesAll({ a: 1, b: 2 }, { a: 1, c: 3 }), false);
  assert.equal(matchesAll({ a: 1 }, {}), true);
});

test('no rules: returns the cartesian product', () => {
  deepEqual(
    buildMatrix({ a: [1, 2] }, []),
    [{ a: 1 }, { a: 2 }],
  );
});

test('extend: matching axis criterion extends only the matched row', () => {
  deepEqual(
    buildMatrix(
      { platform: ['x86_64', 'aarch64'] },
      [{ extend: { platform: 'x86_64', os: 'foo' } }],
    ),
    [
      { platform: 'x86_64', os: 'foo' },
      { platform: 'aarch64' },
    ],
  );
});

test('extend: non-matching axis value is a no-op (never adds)', () => {
  deepEqual(
    buildMatrix(
      { platform: ['x86_64'] },
      [{ extend: { platform: 'aarch64', os: 'foo' } }],
    ),
    [{ platform: 'x86_64' }],
  );
});

test('extend: no axis keys applies extras to every entry', () => {
  deepEqual(
    buildMatrix(
      { a: [1, 2] },
      [{ extend: { tag: 'common' } }],
    ),
    [
      { a: 1, tag: 'common' },
      { a: 2, tag: 'common' },
    ],
  );
});

test('include: matching axis criterion extends matched rows', () => {
  deepEqual(
    buildMatrix(
      { platform: ['x86_64', 'aarch64'] },
      [{ include: { platform: 'x86_64', os: 'foo' } }],
    ),
    [
      { platform: 'x86_64', os: 'foo' },
      { platform: 'aarch64' },
    ],
  );
});

test('include: non-matching axis value appends a new entry', () => {
  deepEqual(
    buildMatrix(
      { platform: ['x86_64'] },
      [{ include: { platform: 'aarch64', os: 'foo' } }],
    ),
    [
      { platform: 'x86_64' },
      { platform: 'aarch64', os: 'foo' },
    ],
  );
});

test('include: no axis keys appends unconditionally', () => {
  deepEqual(
    buildMatrix(
      { a: [1] },
      [{ include: { extraField: 'x' } }],
    ),
    [
      { a: 1 },
      { extraField: 'x' },
    ],
  );
});

test('exclude: drops entries matching the subset', () => {
  deepEqual(
    buildMatrix(
      { a: [1, 2], b: [3, 4] },
      [{ exclude: { a: 1, b: 3 } }],
    ),
    [
      { a: 1, b: 4 },
      { a: 2, b: 3 },
      { a: 2, b: 4 },
    ],
  );
});

test('if: false / "false" / "0" / "" skip the rule', () => {
  for (const cond of [false, 'false', '0', '', 'no', 'off', null]) {
    deepEqual(
      buildMatrix(
        { a: [1, 2] },
        [{ if: cond, exclude: { a: 1 } }],
      ),
      [{ a: 1 }, { a: 2 }],
      `if=${JSON.stringify(cond)} should skip`,
    );
  }
});

test('if: true / "true" / "1" / "yes" run the rule', () => {
  for (const cond of [true, 'true', '1', 'yes', 'on', 'anything']) {
    deepEqual(
      buildMatrix(
        { a: [1, 2] },
        [{ if: cond, exclude: { a: 1 } }],
      ),
      [{ a: 2 }],
      `if=${JSON.stringify(cond)} should run`,
    );
  }
});

test('ordering: extend then exclude removes; exclude then extend leaves row gone', () => {
  // extend then exclude: the row gets extended, then removed.
  deepEqual(
    buildMatrix(
      { p: ['a', 'b'] },
      [
        { extend: { p: 'a', tag: 'X' } },
        { exclude: { p: 'a' } },
      ],
    ),
    [{ p: 'b' }],
  );

  // exclude then extend of the same axis-value: a is gone before extend tries
  // to attach fields to it, so extend is a no-op.
  deepEqual(
    buildMatrix(
      { p: ['a', 'b'] },
      [
        { exclude: { p: 'a' } },
        { extend: { p: 'a', tag: 'X' } },
      ],
    ),
    [{ p: 'b' }],
  );
});

test('ordering: exclude then include of the same axis-value re-adds the row', () => {
  // This is the case GitHub's native include/exclude cannot model: include
  // applied after exclude in source order resurrects the row.
  deepEqual(
    buildMatrix(
      { p: ['a', 'b'] },
      [
        { exclude: { p: 'a' } },
        { include: { p: 'a', tag: 'fresh' } },
      ],
    ),
    [
      { p: 'b' },
      { p: 'a', tag: 'fresh' },
    ],
  );
});

test('empty matrix + rules produces only what rules add', () => {
  deepEqual(
    buildMatrix(
      {},
      [
        { include: { p: 'a', runner: 'x' } },
        { include: { p: 'b', runner: 'y' } },
      ],
    ),
    [
      { p: 'a', runner: 'x' },
      { p: 'b', runner: 'y' },
    ],
  );
});

test('error: matrix must be a map or a list of entries', () => {
  assert.throws(() => buildMatrix(null, []), /must be a map/);
  assert.throws(() => buildMatrix('foo', []), /must be a map/);
  assert.throws(() => buildMatrix(42, []), /must be a map/);
});

test('error: rules must be a list', () => {
  assert.throws(() => buildMatrix({}, {}), /must be a list/);
});

test('error: rule must have exactly one of include/extend/exclude', () => {
  assert.throws(
    () => buildMatrix({}, [{ include: { a: 1 }, exclude: { a: 1 } }]),
    /exactly one of/,
  );
  assert.throws(
    () => buildMatrix({}, [{ if: true }]),
    /exactly one of/,
  );
});

test('error: rule body must be an object or a list of objects', () => {
  assert.throws(
    () => buildMatrix({}, [{ include: 'not-an-object' }]),
    /'include' must be an object or a list of objects/,
  );
});

test('error: list-body entry must be an object', () => {
  assert.throws(
    () => buildMatrix({}, [{ extend: [{ a: 1 }, 'not-an-object'] }]),
    /'extend' entry #1 must be an object/,
  );
});

test('list body: extend applies each entry in order', () => {
  deepEqual(
    buildMatrix(
      { p: ['a', 'b', 'c'] },
      [
        {
          extend: [
            { p: 'a', tag: 'A' },
            { p: 'b', tag: 'B' },
            { p: 'c', tag: 'C' },
          ],
        },
      ],
    ),
    [
      { p: 'a', tag: 'A' },
      { p: 'b', tag: 'B' },
      { p: 'c', tag: 'C' },
    ],
  );
});

test('list body: include applies each entry in order (add + extend)', () => {
  deepEqual(
    buildMatrix(
      { p: ['a'] },
      [
        {
          include: [
            { p: 'a', tag: 'A' },
            { p: 'b', tag: 'B' },
          ],
        },
      ],
    ),
    [
      { p: 'a', tag: 'A' },
      { p: 'b', tag: 'B' },
    ],
  );
});

test('list body: exclude drops each match in order', () => {
  deepEqual(
    buildMatrix(
      { p: ['a', 'b', 'c'] },
      [{ exclude: [{ p: 'a' }, { p: 'c' }] }],
    ),
    [{ p: 'b' }],
  );
});

test('list body: empty list is a no-op', () => {
  deepEqual(
    buildMatrix({ p: ['a', 'b'] }, [{ extend: [] }]),
    [{ p: 'a' }, { p: 'b' }],
  );
});

test('list body: if-guard applies to the whole list', () => {
  deepEqual(
    buildMatrix(
      { p: ['a', 'b'] },
      [
        {
          if: false,
          extend: [
            { p: 'a', tag: 'A' },
            { p: 'b', tag: 'B' },
          ],
        },
      ],
    ),
    [{ p: 'a' }, { p: 'b' }],
  );
});

test('list body: order within the list matters', () => {
  // Second entry shadows the first because both target the same row.
  deepEqual(
    buildMatrix(
      { p: ['a'] },
      [
        {
          extend: [
            { p: 'a', tag: 'first' },
            { p: 'a', tag: 'second' },
          ],
        },
      ],
    ),
    [{ p: 'a', tag: 'second' }],
  );
});

// Parity snapshot: feed the exact pip-build.yml inputs through buildMatrix()
// for all four (disable_x64, disable_arm64) combinations and compare against
// the JSON the current jq pipeline produces.

const PLATFORMS = ['x86_64', 'aarch64'];
const PY_VERSIONS = ['3.8', '3.9', '3.10', '3.11', '3.12', '3.13', '3.14'];

// Use the grouped list-of-bodies form to exercise that path against the
// captured jq fixtures.
const PLATFORM_RULES = {
  extend: [
    { platform: 'x86_64', 'container-options': '--user root', runner: 'ubuntu-latest', compiler: '/usr/bin/clang++', 'container-prefix': '' },
    { platform: 'aarch64', 'container-options': ' ', runner: 'ubuntu-24.04-arm', compiler: '/usr/bin/clang++', 'container-prefix': 'arm64v8/' },
  ],
};
const PY_RULES = {
  extend: [
    { 'py-version': '3.8', os: 'rockylinux:8' },
    { 'py-version': '3.9', os: 'debian:11-slim' },
    { 'py-version': '3.10', os: 'ubuntu:22.04' },
    { 'py-version': '3.11', os: 'fedora:37' },
    { 'py-version': '3.12', os: 'fedora:39' },
    { 'py-version': '3.13', os: 'fedora:42' },
    { 'py-version': '3.14', os: 'ubuntu:25.10' },
  ],
};

function pipBuildMatrix(disableX64, disableArm64) {
  return buildMatrix(
    { platform: PLATFORMS },
    [
      PLATFORM_RULES,
      {
        extend: [
          { platform: 'x86_64', os: 'rockylinux8-vcpkg-x64' },
          { platform: 'aarch64', os: 'rockylinux8-vcpkg-arm64' },
        ],
      },
      { if: disableX64, exclude: { platform: 'x86_64' } },
      { if: disableArm64, exclude: { platform: 'aarch64' } },
    ],
  );
}

function pipTestMatrix(disableX64, disableArm64) {
  return buildMatrix(
    { platform: PLATFORMS, 'py-version': PY_VERSIONS },
    [
      PLATFORM_RULES,
      PY_RULES,
      { if: disableX64, exclude: { platform: 'x86_64' } },
      { if: disableArm64, exclude: { platform: 'aarch64' } },
    ],
  );
}

function loadFixture(name) {
  const p = path.join(__dirname, 'fixtures', name);
  return JSON.parse(fs.readFileSync(p, 'utf8'));
}

const COMBOS = [
  { suffix: '00', dx: false, da: false },
  { suffix: '01', dx: false, da: true },
  { suffix: '10', dx: true, da: false },
  { suffix: '11', dx: true, da: true },
];

for (const { suffix, dx, da } of COMBOS) {
  test(`parity: pip-build build matrix (disable_x64=${dx}, disable_arm64=${da})`, () => {
    deepEqual(pipBuildMatrix(dx, da), loadFixture(`build-${suffix}.json`));
  });
  test(`parity: pip-build test matrix (disable_x64=${dx}, disable_arm64=${da})`, () => {
    deepEqual(pipTestMatrix(dx, da), loadFixture(`test-${suffix}.json`));
  });
}

// ---------------------------------------------------------------------------
// List-form base matrix: include-style entries taken verbatim, axis keys =
// union of the entries' keys.

test('list base: entries pass through verbatim with no rules', () => {
  deepEqual(
    buildMatrix([{ a: 1, b: 'x' }, { a: 2 }], []),
    [{ a: 1, b: 'x' }, { a: 2 }],
  );
});

test('list base: empty list yields an empty matrix', () => {
  deepEqual(buildMatrix([], []), []);
});

test('list base: extend matches on entry keys', () => {
  deepEqual(
    buildMatrix(
      [{ distro: 'ubuntu22', arch: 'x64' }, { distro: 'ubuntu22', arch: 'arm64' }],
      [{ extend: { arch: 'arm64', runner: 'ubuntu-24.04-arm' } }],
    ),
    [
      { distro: 'ubuntu22', arch: 'x64' },
      { distro: 'ubuntu22', arch: 'arm64', runner: 'ubuntu-24.04-arm' },
    ],
  );
});

test('list base: axis keys are the union of entry keys', () => {
  // `runner` appears in only one base entry but is still an axis key, so an
  // include matching on it extends that entry instead of appending a new one.
  deepEqual(
    buildMatrix(
      [{ a: 1 }, { a: 2, runner: 'r1' }],
      [{ include: { runner: 'r1', tag: 't' } }],
    ),
    [{ a: 1 }, { a: 2, runner: 'r1', tag: 't' }],
  );
});

test('list base: include with unmatched axis criteria appends', () => {
  deepEqual(
    buildMatrix(
      [{ distro: 'ubuntu22', arch: 'x64' }],
      [{ include: { distro: 'ubuntu24', arch: 'x64' } }],
    ),
    [{ distro: 'ubuntu22', arch: 'x64' }, { distro: 'ubuntu24', arch: 'x64' }],
  );
});

test('list base: exclude drops matching entries', () => {
  deepEqual(
    buildMatrix(
      [{ distro: 'ubuntu22', arch: 'x64' }, { distro: 'emscripten', arch: 'arm64' }],
      [{ exclude: { distro: 'emscripten' } }],
    ),
    [{ distro: 'ubuntu22', arch: 'x64' }],
  );
});

test('list base: input entries are not mutated', () => {
  const base = [{ a: 1 }];
  buildMatrix(base, [{ extend: { b: 2 } }]);
  assert.deepStrictEqual(base, [{ a: 1 }]);
});

test('error: list base entry must be an object', () => {
  assert.throws(() => buildMatrix([{ a: 1 }, 'nope'], []), /entry #1 must be an object/);
  assert.throws(() => buildMatrix([[1]], []), /entry #0 must be an object/);
});

// ---------------------------------------------------------------------------
// Parity with the jq pipeline these rules replaced in prepare-images.yml
// (compute-image-matrices). The docker-linux-<dx64><darm64><demscr>.json
// fixtures are the captured outputs of the original jq filter over
// fixtures/docker-images.json (a snapshot of
// .github/workflows/matrix/docker-images.json). The rules below must stay in
// sync with the matrix-builder step in prepare-images.yml.

const DOCKER_IMAGES = loadFixture('docker-images.json');

function dockerLinuxMatrix(dx64, darm64, demscr) {
  return buildMatrix(DOCKER_IMAGES.linux, [
    {
      if: dx64,
      exclude: [
        { distro: 'ubuntu22', arch: 'x64' },
        { distro: 'ubuntu24', arch: 'x64' },
      ],
    },
    {
      if: darm64,
      exclude: [
        { distro: 'ubuntu22', arch: 'arm64' },
        { distro: 'ubuntu24', arch: 'arm64' },
      ],
    },
    { if: demscr, exclude: { distro: 'emscripten' } },
  ]);
}

for (const dx64 of [false, true]) {
  for (const darm64 of [false, true]) {
    for (const demscr of [false, true]) {
      const suffix = `${+dx64}${+darm64}${+demscr}`;
      test(`parity: prepare-images linux matrix (${suffix})`, () => {
        deepEqual(dockerLinuxMatrix(dx64, darm64, demscr), loadFixture(`docker-linux-${suffix}.json`));
      });
    }
  }
}
