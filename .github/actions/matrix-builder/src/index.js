'use strict';

const fs = require('node:fs');
const core = require('@actions/core');
const yaml = require('js-yaml');
const { buildMatrix } = require('../lib/engine');

/**
 * Read an action input as YAML/JSON. Empty / missing returns the fallback.
 *
 * Uses the YAML JSON_SCHEMA so YAML 1.1 booleans (`yes`/`no`/`on`/`off`) are
 * not silently coerced - they remain strings, matching what workflow authors
 * see when they read their own YAML.
 *
 * @param {string} name Input name as declared in action.yml.
 * @param {*} fallback Value returned when the input is empty.
 * @returns {*} Parsed structure.
 * @throws {Error} If the input is non-empty but fails to parse.
 */
function parseInput(name, fallback) {
  const raw = core.getInput(name);
  if (!raw || raw.trim() === '') return fallback;
  try {
    return yaml.load(raw, { schema: yaml.JSON_SCHEMA });
  } catch (e) {
    throw new Error(`Failed to parse input '${name}' as YAML/JSON: ${e.message}`);
  }
}

/**
 * Load the base matrix from a YAML/JSON file.
 *
 * @param {string} file Path as given, resolved against the process cwd
 *   (the workspace root when run by GitHub).
 * @returns {*} The base matrix structure.
 * @throws {Error} If the file is unreadable or unparsable.
 */
function loadMatrixFile(file) {
  let raw;
  try {
    raw = fs.readFileSync(file, 'utf8');
  } catch (e) {
    throw new Error(`Failed to read matrix file '${file}': ${e.message}`);
  }
  let doc;
  try {
    doc = yaml.load(raw, { schema: yaml.JSON_SCHEMA });
  } catch (e) {
    throw new Error(`Failed to parse matrix file '${file}' as YAML/JSON: ${e.message}`);
  }
  return doc == null ? {} : doc;
}

try {
  const matrixFile = core.getInput('matrix-file');
  if (matrixFile && core.getInput('matrix')) {
    throw new Error("'matrix' and 'matrix-file' are mutually exclusive");
  }
  const matrix = matrixFile
    ? loadMatrixFile(matrixFile)
    : parseInput('matrix', {});
  const rules = parseInput('rules', []);
  const result = buildMatrix(matrix, rules);
  core.setOutput('matrix', JSON.stringify(result));
  core.startGroup(`matrix-builder: ${result.length} combination(s)`);
  core.info(JSON.stringify(result, null, 2));
  core.endGroup();
} catch (e) {
  core.setFailed(e.message);
}
