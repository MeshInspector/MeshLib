'use strict';

/**
 * Build the cartesian product of axis-name -> value-list.
 *
 * A scalar axis value is treated as a one-element list. An empty axes object
 * yields an empty result (no base entries).
 *
 * @param {Object<string, *|Array<*>>} axes
 *   Map of axis name to a list of values (or a scalar).
 * @returns {Array<Object>} Every combination, in lexicographic axis order.
 *
 * @example
 * cartesian({ a: [1, 2], b: ['x', 'y'] });
 * // => [ {a:1,b:'x'}, {a:1,b:'y'}, {a:2,b:'x'}, {a:2,b:'y'} ]
 */
function cartesian(axes) {
  const names = Object.keys(axes);
  if (names.length === 0) return [];
  let acc = [{}];
  for (const name of names) {
    const vals = Array.isArray(axes[name]) ? axes[name] : [axes[name]];
    const next = [];
    for (const entry of acc) {
      for (const v of vals) next.push({ ...entry, [name]: v });
    }
    acc = next;
  }
  return acc;
}

/**
 * Interpret a value as truthy or falsy under this action's rules.
 *
 * GitHub expands `${{ ... }}` expressions before the step runs, so the action
 * generally sees `if:` as a string. The truthiness rules:
 *   - YAML `true` => truthy; `false` / `null` / `undefined` => falsy.
 *   - Strings: trim + lowercase; `""`, `"false"`, `"0"`, `"no"`, `"off"` =>
 *     falsy; everything else => truthy.
 *   - Numbers: standard JS truthiness.
 *
 * @param {*} v Any value.
 * @returns {boolean}
 */
function isTruthy(v) {
  if (v === true) return true;
  if (v === false || v == null) return false;
  if (typeof v === 'string') {
    const s = v.trim().toLowerCase();
    return !(s === '' || s === 'false' || s === '0' || s === 'no' || s === 'off');
  }
  return Boolean(v);
}

/**
 * Return true iff `entry` contains every key in `criteria` with the same value.
 *
 * Keys present in `criteria` but absent from `entry` cause a non-match. This
 * mirrors GitHub's own matrix subset-match semantics.
 *
 * @param {Object} entry
 * @param {Object} criteria
 * @returns {boolean}
 */
function matchesAll(entry, criteria) {
  for (const k of Object.keys(criteria)) {
    if (!Object.prototype.hasOwnProperty.call(entry, k)) return false;
    if (entry[k] !== criteria[k]) return false;
  }
  return true;
}

/**
 * Partition a rule body into axis-key criteria and extra-key payload.
 *
 * Axis keys are those that exist in the base matrix's axis list. Everything
 * else is treated as a payload (extra) field merged into matched entries.
 *
 * @param {Object} rule
 * @param {Set<string>} axisKeys
 * @returns {{ axisCrit: Object, extra: Object }}
 */
function splitRule(rule, axisKeys) {
  const axisCrit = {};
  const extra = {};
  for (const [k, v] of Object.entries(rule)) {
    if (axisKeys.has(k)) axisCrit[k] = v;
    else extra[k] = v;
  }
  return { axisCrit, extra };
}

/**
 * Apply an `include` rule (GitHub-native hybrid semantics).
 *
 *   1. Split into axis criteria + extra payload.
 *   2. For every existing entry whose values satisfy the axis criteria, merge
 *      the extra payload into that entry.
 *   3. If no entry matched and the rule has at least one axis key, append a
 *      new entry with all of the rule's keys.
 *   4. If the rule has no axis keys at all, append a new entry unconditionally
 *      (same as GitHub).
 *
 * @param {Array<Object>} entries
 * @param {Object} rule
 * @param {Set<string>} axisKeys
 * @returns {Array<Object>} A new entries list (input is not mutated).
 */
function applyInclude(entries, rule, axisKeys) {
  const { axisCrit, extra } = splitRule(rule, axisKeys);
  // No axis keys at all: GitHub-native semantics say to append a new entry
  // unconditionally rather than extending every existing entry. (Without this
  // special case, the empty-axisCrit subset would match every row and the
  // rule would silently extend everything.)
  if (Object.keys(axisCrit).length === 0) {
    return [...entries, { ...extra }];
  }
  let matched = 0;
  const out = entries.map(e => {
    if (matchesAll(e, axisCrit)) {
      matched++;
      return { ...e, ...extra };
    }
    return e;
  });
  if (matched === 0) out.push({ ...axisCrit, ...extra });
  return out;
}

/**
 * Apply an `extend` rule (strict extend-only; never appends).
 *
 *   1. Split into axis criteria + extra payload.
 *   2. For every existing entry whose values satisfy the axis criteria, merge
 *      the extra payload into that entry.
 *   3. If no entry matched, do nothing (silent no-op).
 *   4. With no axis keys, every entry trivially matches, so the extra payload
 *      is applied to all entries.
 *
 * @param {Array<Object>} entries
 * @param {Object} rule
 * @param {Set<string>} axisKeys
 * @returns {Array<Object>} A new entries list (input is not mutated).
 */
function applyExtend(entries, rule, axisKeys) {
  const { axisCrit, extra } = splitRule(rule, axisKeys);
  return entries.map(e => matchesAll(e, axisCrit) ? { ...e, ...extra } : e);
}

/**
 * Apply an `exclude` rule (subset match drop).
 *
 * Remove every entry where, for each key in the rule, the entry has that key
 * with the matching value. Keys not present in an entry don't match (so they
 * survive) - same as GitHub's native matrix.exclude.
 *
 * @param {Array<Object>} entries
 * @param {Object} rule
 * @returns {Array<Object>} A new entries list (input is not mutated).
 */
function applyExclude(entries, rule) {
  return entries.filter(e => !matchesAll(e, rule));
}

/**
 * Normalize an `include` / `extend` / `exclude` body into a list of one or
 * more rule bodies. Both shapes are accepted:
 *
 *   extend:
 *     platform: x86_64
 *     os: rockylinux8
 *
 *   extend:
 *     - platform: x86_64
 *       os: rockylinux8
 *     - platform: aarch64
 *       os: rockylinux8-arm64
 *
 * The single-object form is wrapped in a one-element list. The list form is
 * returned as-is after validating each entry is an object. An empty list is
 * legal (it's a no-op once iterated over).
 *
 * @param {*} body Raw body value as parsed from the action input.
 * @param {string} kind The keyword that introduced it (for error messages).
 * @param {number} ruleIndex Index of the parent rule (for error messages).
 * @returns {Array<Object>}
 * @throws {Error} If the body is neither an object nor a list of objects.
 */
function normalizeBodies(body, kind, ruleIndex) {
  if (Array.isArray(body)) {
    body.forEach((b, j) => {
      if (b == null || typeof b !== 'object' || Array.isArray(b)) {
        throw new Error(`rule #${ruleIndex} '${kind}' entry #${j} must be an object`);
      }
    });
    return body;
  }
  if (body == null || typeof body !== 'object') {
    throw new Error(`rule #${ruleIndex} '${kind}' must be an object or a list of objects`);
  }
  return [body];
}

/**
 * Build a GitHub Actions matrix from a base axis-map and an ordered list of
 * rules.
 *
 * Rules are applied strictly in source order; each rule sees the state
 * produced by all prior rules. This differs from GitHub's native matrix
 * where `exclude` runs first and `include` runs unconditionally after - so
 * this engine can model order-dependent flows that native matrix syntax
 * cannot.
 *
 * @param {Object<string, *|Array<*>>} baseMatrix
 *   Axis name -> value list. Scalars are treated as one-element lists. Pass
 *   `{}` (or omit) to start from an empty entry list.
 * @param {Array<Object>} rules
 *   Ordered rules. Each rule must have exactly one of `include`, `extend`,
 *   `exclude`, optionally guarded by `if`.
 * @returns {Array<Object>} The resulting matrix combinations.
 * @throws {Error} If inputs are malformed.
 *
 * @example
 * buildMatrix(
 *   { platform: ['x86_64', 'aarch64'] },
 *   [
 *     { extend:  { platform: 'x86_64', runner: 'ubuntu-latest' } },
 *     { exclude: { platform: 'aarch64' } },
 *   ]
 * );
 * // => [ { platform: 'x86_64', runner: 'ubuntu-latest' } ]
 */
function buildMatrix(baseMatrix, rules) {
  if (baseMatrix == null || typeof baseMatrix !== 'object' || Array.isArray(baseMatrix)) {
    throw new Error("'matrix' must be a map of axis names to value lists");
  }
  if (!Array.isArray(rules)) {
    throw new Error("'rules' must be a list");
  }
  const axisKeys = new Set(Object.keys(baseMatrix));
  let entries = cartesian(baseMatrix);

  rules.forEach((rule, i) => {
    if (rule == null || typeof rule !== 'object' || Array.isArray(rule)) {
      throw new Error(`rule #${i} is not an object`);
    }
    if ('if' in rule && !isTruthy(rule.if)) return;
    const kinds = ['include', 'extend', 'exclude'].filter(k => k in rule);
    if (kinds.length !== 1) {
      throw new Error(
        `rule #${i} must have exactly one of 'include', 'extend', 'exclude' ` +
        `(found: ${kinds.length ? kinds.join(', ') : 'none'})`
      );
    }
    const kind = kinds[0];
    const bodies = normalizeBodies(rule[kind], kind, i);
    for (const body of bodies) {
      if (kind === 'include') entries = applyInclude(entries, body, axisKeys);
      else if (kind === 'extend') entries = applyExtend(entries, body, axisKeys);
      else entries = applyExclude(entries, body);
    }
  });

  return entries;
}

module.exports = {
  buildMatrix,
  cartesian,
  isTruthy,
  matchesAll,
  applyInclude,
  applyExtend,
  applyExclude,
  normalizeBodies,
};
