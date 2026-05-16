'use strict';

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

try {
  const matrix = parseInput('matrix', {});
  const rules = parseInput('rules', []);
  const result = buildMatrix(matrix, rules);
  core.setOutput('matrix', JSON.stringify(result));
  core.startGroup(`matrix-builder: ${result.length} combination(s)`);
  core.info(JSON.stringify(result, null, 2));
  core.endGroup();
} catch (e) {
  core.setFailed(e.message);
}
