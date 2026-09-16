"""
Build a GitHub Actions matrix from a base matrix and an ordered list of
include / extend / exclude rules (see README.md for the grammar).

The base matrix is read as JSON, either an axis map or a list of entries,
from the file named by the positional argument, or from stdin when that
argument is '-'; the rules arrive as a JSON list in --rules. The action step
produces both with yq:

    MATRIX=$(yq -o json . <<<"$MATRIX")
    RULES=$(yq -o json . <<<"$RULES")
    echo "$MATRIX" | python3 matrix_builder.py --rules="$RULES" -

The resulting combinations are appended to the file named by GITHUB_OUTPUT
as `matrix=<json>`.
"""

import argparse
import json
import os
import sys


def cartesian(axes):
    if not axes:
        return []
    entries = [{}]
    for name, values in axes.items():
        if not isinstance(values, list):
            values = [values]
        entries = [{**entry, name: value} for entry in entries for value in values]
    return entries


def is_truthy(value):
    if isinstance(value, str):
        return value.strip().lower() not in ('', 'false', '0', 'no', 'off')
    return bool(value)


def same_value(a, b):
    return isinstance(a, bool) == isinstance(b, bool) and a == b


def matches_all(entry, criteria):
    return all(key in entry and same_value(entry[key], value) for key, value in criteria.items())


def split_rule(rule, axis_keys):
    axis_crit = {k: v for k, v in rule.items() if k in axis_keys}
    extra = {k: v for k, v in rule.items() if k not in axis_keys}
    return axis_crit, extra


def apply_include(entries, rule, axis_keys):
    axis_crit, extra = split_rule(rule, axis_keys)
    if not axis_crit:
        return [*entries, dict(extra)]
    matched = [matches_all(entry, axis_crit) for entry in entries]
    out = [{**entry, **extra} if hit else entry for entry, hit in zip(entries, matched)]
    if not any(matched):
        out.append({**axis_crit, **extra})
    return out


def apply_extend(entries, rule, axis_keys):
    axis_crit, extra = split_rule(rule, axis_keys)
    return [{**entry, **extra} if matches_all(entry, axis_crit) else entry for entry in entries]


def apply_exclude(entries, rule):
    return [entry for entry in entries if not matches_all(entry, rule)]


def normalize_bodies(body, kind, rule_index):
    if isinstance(body, list):
        for j, item in enumerate(body):
            if not isinstance(item, dict):
                raise ValueError(f"rule #{rule_index} '{kind}' entry #{j} must be an object")
        return body
    if not isinstance(body, dict):
        raise ValueError(f"rule #{rule_index} '{kind}' must be an object or a list of objects")
    return [body]


def build_matrix(base_matrix, rules):
    if isinstance(base_matrix, list):
        for i, entry in enumerate(base_matrix):
            if not isinstance(entry, dict):
                raise ValueError(f'base matrix entry #{i} must be an object')
        axis_keys = {key for entry in base_matrix for key in entry}
        entries = [dict(entry) for entry in base_matrix]
    elif isinstance(base_matrix, dict):
        axis_keys = set(base_matrix)
        entries = cartesian(base_matrix)
    else:
        raise ValueError("'matrix' must be a map of axis names to value lists or a list of entries")
    if not isinstance(rules, list):
        raise ValueError("'rules' must be a list")

    for i, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError(f'rule #{i} is not an object')
        if 'if' in rule and not is_truthy(rule['if']):
            continue
        kinds = [kind for kind in ('include', 'extend', 'exclude') if kind in rule]
        if len(kinds) != 1:
            found = ', '.join(kinds) if kinds else 'none'
            raise ValueError(
                f"rule #{i} must have exactly one of 'include', 'extend', 'exclude' (found: {found})"
            )
        kind = kinds[0]
        for body in normalize_bodies(rule[kind], kind, i):
            if kind == 'include':
                entries = apply_include(entries, body, axis_keys)
            elif kind == 'extend':
                entries = apply_extend(entries, body, axis_keys)
            else:
                entries = apply_exclude(entries, body)
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('matrix', type=argparse.FileType(), help="JSON base matrix file, or '-' for stdin")
    parser.add_argument('--rules', required=True, help='JSON list of rules, applied in order')
    args = parser.parse_args()
    with args.matrix as f:
        matrix = json.load(f)
    rules = json.loads(args.rules)
    try:
        result = build_matrix({} if matrix is None else matrix, [] if rules is None else rules)
    except ValueError as e:
        print(f'::error::{e}')
        return 1
    with open(os.environ['GITHUB_OUTPUT'], 'a') as out:
        print(f"matrix={json.dumps(result, separators=(',', ':'))}", file=out)
    print(f'::group::matrix-builder: {len(result)} combination(s)')
    print(json.dumps(result, indent=2))
    print('::endgroup::')
    return 0


if __name__ == '__main__':
    sys.exit(main())
