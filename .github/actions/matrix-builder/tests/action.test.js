'use strict';

// Input-handling tests: run src/index.js as a child process the way GitHub
// does (inputs come in as INPUT_* env vars, outputs go to $GITHUB_OUTPUT).
// Requires node_modules (npm ci) since src/index.js pulls in @actions/core.

const test = require('node:test');
const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const ACTION_DIR = path.join(__dirname, '..');
const ENTRY = path.join(ACTION_DIR, 'src', 'index.js');
const FIXTURES = path.join(__dirname, 'fixtures');

// Parse the GITHUB_OUTPUT file format written by @actions/core:
//   name<<DELIMITER
//   value (possibly multi-line)
//   DELIMITER
function parseGithubOutput(text) {
  const out = {};
  const lines = text.replace(/\r\n/g, '\n').split('\n');
  for (let i = 0; i < lines.length; i++) {
    const m = /^(.+?)<<(.+)$/.exec(lines[i]);
    if (!m) continue;
    const val = [];
    for (i++; i < lines.length && lines[i] !== m[2]; i++) val.push(lines[i]);
    out[m[1]] = val.join('\n');
  }
  return out;
}

/**
 * Run the action entry point with the given inputs.
 *
 * @param {Object<string, string>} inputs Input name -> value (as YAML text).
 * @returns {{status: number, stdout: string, outputs: Object<string, string>}}
 */
function runAction(inputs) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'matrix-builder-test-'));
  const outFile = path.join(dir, 'github_output');
  fs.writeFileSync(outFile, '');
  const env = { ...process.env, GITHUB_OUTPUT: outFile };
  for (const [name, value] of Object.entries(inputs)) {
    env[`INPUT_${name.toUpperCase()}`] = value;
  }
  const res = spawnSync(process.execPath, [ENTRY], {
    env,
    cwd: ACTION_DIR,
    encoding: 'utf8',
  });
  const outputs = parseGithubOutput(fs.readFileSync(outFile, 'utf8'));
  fs.rmSync(dir, { recursive: true, force: true });
  return { status: res.status, stdout: res.stdout, outputs };
}

function matrixOf(result) {
  assert.equal(result.status, 0, `action failed: ${result.stdout}`);
  return JSON.parse(result.outputs.matrix);
}

test('matrix-file: JSON file with matrix-key selects one bundled matrix', () => {
  const result = runAction({
    'matrix-file': path.join(FIXTURES, 'docker-images.json'),
    'matrix-key': 'linux',
    rules: [
      '- if: true',
      '  exclude:',
      '    - { distro: ubuntu22, arch: x64 }',
      '    - { distro: ubuntu24, arch: x64 }',
    ].join('\n'),
  });
  const expected = JSON.parse(
    fs.readFileSync(path.join(FIXTURES, 'docker-images.json'), 'utf8'),
  ).linux.filter(e => e.arch !== 'x64');
  assert.deepStrictEqual(matrixOf(result), expected);
});

test('matrix-file: workspace-relative path is resolved against cwd', () => {
  const result = runAction({
    'matrix-file': path.join('tests', 'fixtures', 'docker-images.json'),
    'matrix-key': 'linux-vcpkg',
  });
  assert.deepStrictEqual(matrixOf(result), [
    { os: 'rockylinux8', arch: 'x64', runner: 'ubuntu-latest' },
    { os: 'rockylinux8', arch: 'arm64', runner: 'ubuntu-24.04-arm' },
  ]);
});

test('matrix-file: YAML axis map without matrix-key', () => {
  const result = runAction({
    'matrix-file': path.join(FIXTURES, 'axes.yml'),
  });
  assert.deepStrictEqual(matrixOf(result), [
    { platform: 'x86_64', config: 'Release' },
    { platform: 'aarch64', config: 'Release' },
  ]);
});

test('matrix-file: YAML entry-list document', () => {
  const result = runAction({
    'matrix-file': path.join(FIXTURES, 'entry-list.yml'),
    rules: '[{ exclude: { arch: arm64 } }]',
  });
  assert.deepStrictEqual(matrixOf(result), [{ distro: 'ubuntu22', arch: 'x64' }]);
});

test('inline matrix still works (list form)', () => {
  const result = runAction({
    matrix: '[{ a: 1 }, { a: 2 }]',
    rules: '[{ extend: { a: 2, b: 3 } }]',
  });
  assert.deepStrictEqual(matrixOf(result), [{ a: 1 }, { a: 2, b: 3 }]);
});

function assertFails(result, pattern) {
  assert.equal(result.status, 1);
  assert.match(result.stdout, pattern);
}

test("error: 'matrix' and 'matrix-file' are mutually exclusive", () => {
  assertFails(
    runAction({
      matrix: 'platform: [x86_64]',
      'matrix-file': path.join(FIXTURES, 'axes.yml'),
    }),
    /mutually exclusive/,
  );
});

test("error: 'matrix-key' requires 'matrix-file'", () => {
  assertFails(runAction({ 'matrix-key': 'linux' }), /'matrix-key' requires 'matrix-file'/);
});

test('error: matrix-file does not exist', () => {
  assertFails(
    runAction({ 'matrix-file': path.join(FIXTURES, 'no-such-file.json') }),
    /Failed to read matrix file/,
  );
});

test('error: matrix-key absent from the document', () => {
  assertFails(
    runAction({
      'matrix-file': path.join(FIXTURES, 'docker-images.json'),
      'matrix-key': 'windows',
    }),
    /has no top-level key 'windows'/,
  );
});

test('error: matrix-key on a non-map document', () => {
  assertFails(
    runAction({
      'matrix-file': path.join(FIXTURES, 'entry-list.yml'),
      'matrix-key': 'linux',
    }),
    /must be a map to select 'matrix-key' from/,
  );
});
