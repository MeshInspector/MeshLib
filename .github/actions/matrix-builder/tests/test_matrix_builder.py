import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ACTION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ACTION_DIR))

from matrix_builder import build_matrix, cartesian, is_truthy, matches_all  # noqa: E402

CIRCLE_RED = {'shape': 'circle', 'color': 'red'}
CIRCLE_BLUE = {'shape': 'circle', 'color': 'blue'}
SQUARE_RED = {'shape': 'square', 'color': 'red'}
SQUARE_BLUE = {'shape': 'square', 'color': 'blue'}
SHAPE_COLOR = {'shape': ['circle', 'square'], 'color': ['red', 'blue']}


def read_outputs(path):
    return dict(line.split('=', 1) for line in path.read_text().splitlines())


class CartesianTest(unittest.TestCase):
    def test_no_axes_gives_no_entries(self):
        self.assertEqual(cartesian({}), [])

    def test_single_axis(self):
        self.assertEqual(cartesian({'size': ['small', 'large']}), [{'size': 'small'}, {'size': 'large'}])

    def test_first_axis_varies_slowest(self):
        self.assertEqual(cartesian(SHAPE_COLOR), [CIRCLE_RED, CIRCLE_BLUE, SQUARE_RED, SQUARE_BLUE])

    def test_scalar_axis_value_counts_as_one_element_list(self):
        self.assertEqual(
            cartesian({'shape': 'circle', 'size': ['small', 'large']}),
            [{'shape': 'circle', 'size': 'small'}, {'shape': 'circle', 'size': 'large'}],
        )


class TruthinessTest(unittest.TestCase):
    def test_literals(self):
        self.assertTrue(is_truthy(True))
        self.assertFalse(is_truthy(False))
        self.assertFalse(is_truthy(None))

    def test_numbers(self):
        self.assertFalse(is_truthy(0))
        self.assertTrue(is_truthy(1))
        self.assertTrue(is_truthy(2.5))

    def test_falsy_strings_ignore_case_and_whitespace(self):
        for s in ['', '   ', 'false', 'FALSE', ' False ', '0', 'no', 'NO', 'off', 'Off']:
            with self.subTest(s=repr(s)):
                self.assertFalse(is_truthy(s))

    def test_other_strings_are_truthy(self):
        for s in ['true', '1', 'yes', 'on', 'maybe', ' x ', 'null']:
            with self.subTest(s=repr(s)):
                self.assertTrue(is_truthy(s))


class MatchesAllTest(unittest.TestCase):
    def test_criteria_are_a_subset_of_the_entry(self):
        entry = {'shape': 'circle', 'color': 'red', 'sides': 0}
        self.assertTrue(matches_all(entry, {}))
        self.assertTrue(matches_all(entry, {'shape': 'circle'}))
        self.assertTrue(matches_all(entry, {'shape': 'circle', 'color': 'red'}))
        self.assertFalse(matches_all(entry, {'shape': 'circle', 'color': 'blue'}))

    def test_missing_key_never_matches(self):
        self.assertFalse(matches_all({'shape': 'circle'}, {'shape': 'circle', 'sides': None}))

    def test_bool_and_int_are_distinct(self):
        self.assertFalse(matches_all({'solid': True}, {'solid': 1}))
        self.assertFalse(matches_all({'solid': 0}, {'solid': False}))

    def test_int_and_equal_float_match(self):
        self.assertTrue(matches_all({'sides': 4}, {'sides': 4.0}))

    def test_lists_compare_by_value(self):
        self.assertTrue(matches_all({'tags': ['round', 'flat']}, {'tags': ['round', 'flat']}))
        self.assertFalse(matches_all({'tags': ['round', 'flat']}, {'tags': ['flat', 'round']}))


class IncludeTest(unittest.TestCase):
    def test_matching_axis_values_extend_every_matching_row(self):
        result = build_matrix(SHAPE_COLOR, [{'include': {'shape': 'circle', 'sides': 0}}])
        self.assertEqual(result, [
            {**CIRCLE_RED, 'sides': 0},
            {**CIRCLE_BLUE, 'sides': 0},
            SQUARE_RED,
            SQUARE_BLUE,
        ])

    def test_unmatched_axis_values_append_a_row_with_the_rule_keys(self):
        result = build_matrix({'shape': ['circle']}, [{'include': {'shape': 'triangle', 'sides': 3}}])
        self.assertEqual(result, [{'shape': 'circle'}, {'shape': 'triangle', 'sides': 3}])

    def test_no_axis_keys_appends_unconditionally(self):
        result = build_matrix({'shape': ['circle']}, [{'include': {'sides': 3}}])
        self.assertEqual(result, [{'shape': 'circle'}, {'sides': 3}])

    def test_list_body_mixes_extend_and_append(self):
        result = build_matrix({'shape': ['circle']}, [{'include': [
            {'shape': 'circle', 'sides': 0},
            {'shape': 'triangle', 'sides': 3},
        ]}])
        self.assertEqual(result, [{'shape': 'circle', 'sides': 0}, {'shape': 'triangle', 'sides': 3}])

    def test_later_payload_overrides_earlier_one(self):
        result = build_matrix({'shape': ['circle']}, [
            {'include': {'shape': 'circle', 'label': 'disc'}},
            {'include': {'shape': 'circle', 'label': 'ring'}},
        ])
        self.assertEqual(result, [{'shape': 'circle', 'label': 'ring'}])


class ExtendTest(unittest.TestCase):
    def test_matching_axis_values_extend_only_those_rows(self):
        result = build_matrix(SHAPE_COLOR, [{'extend': {'color': 'blue', 'cool': True}}])
        self.assertEqual(result, [
            CIRCLE_RED,
            {**CIRCLE_BLUE, 'cool': True},
            SQUARE_RED,
            {**SQUARE_BLUE, 'cool': True},
        ])

    def test_unmatched_axis_values_are_a_noop(self):
        result = build_matrix({'shape': ['circle']}, [{'extend': {'shape': 'triangle', 'sides': 3}}])
        self.assertEqual(result, [{'shape': 'circle'}])

    def test_no_axis_keys_extends_every_row(self):
        result = build_matrix({'shape': ['circle', 'square']}, [{'extend': {'weight': 1}}])
        self.assertEqual(result, [{'shape': 'circle', 'weight': 1}, {'shape': 'square', 'weight': 1}])

    def test_list_body_applies_each_item_in_order(self):
        result = build_matrix({'shape': ['circle', 'square']}, [{'extend': [
            {'shape': 'circle', 'sides': 0},
            {'shape': 'square', 'sides': 4},
            {'shape': 'square', 'sides': 5},
        ]}])
        self.assertEqual(result, [{'shape': 'circle', 'sides': 0}, {'shape': 'square', 'sides': 5}])

    def test_empty_list_body_is_a_noop(self):
        self.assertEqual(build_matrix({'shape': ['circle']}, [{'extend': []}]), [{'shape': 'circle'}])


class ExcludeTest(unittest.TestCase):
    def test_subset_match_drops_rows(self):
        result = build_matrix(SHAPE_COLOR, [{'exclude': {'shape': 'square', 'color': 'blue'}}])
        self.assertEqual(result, [CIRCLE_RED, CIRCLE_BLUE, SQUARE_RED])

    def test_single_key_drops_every_row_with_that_value(self):
        self.assertEqual(build_matrix(SHAPE_COLOR, [{'exclude': {'color': 'blue'}}]), [CIRCLE_RED, SQUARE_RED])

    def test_key_absent_from_a_row_keeps_that_row(self):
        result = build_matrix([CIRCLE_RED, {'shape': 'triangle'}], [{'exclude': {'color': 'red'}}])
        self.assertEqual(result, [{'shape': 'triangle'}])

    def test_payload_keys_take_part_in_the_match(self):
        result = build_matrix({'shape': ['circle', 'square']}, [
            {'extend': {'shape': 'circle', 'curved': True}},
            {'exclude': {'curved': True}},
        ])
        self.assertEqual(result, [{'shape': 'square'}])

    def test_list_body_drops_each_match(self):
        result = build_matrix(SHAPE_COLOR, [{'exclude': [{'shape': 'circle'}, {'color': 'blue'}]}])
        self.assertEqual(result, [SQUARE_RED])


class IfGuardTest(unittest.TestCase):
    def test_falsy_values_skip_the_rule(self):
        for cond in [False, None, 0, '', 'false', 'FALSE', '0', 'no', 'off', ' false ']:
            with self.subTest(cond=repr(cond)):
                result = build_matrix({'shape': ['circle', 'square']}, [{'if': cond, 'exclude': {'shape': 'square'}}])
                self.assertEqual(result, [{'shape': 'circle'}, {'shape': 'square'}])

    def test_truthy_values_apply_the_rule(self):
        for cond in [True, 1, 'true', 'TRUE', '1', 'yes', 'on', 'maybe']:
            with self.subTest(cond=repr(cond)):
                result = build_matrix({'shape': ['circle', 'square']}, [{'if': cond, 'exclude': {'shape': 'square'}}])
                self.assertEqual(result, [{'shape': 'circle'}])

    def test_absent_if_applies_the_rule(self):
        result = build_matrix({'shape': ['circle', 'square']}, [{'exclude': {'shape': 'square'}}])
        self.assertEqual(result, [{'shape': 'circle'}])

    def test_guard_covers_the_whole_list_body(self):
        result = build_matrix(SHAPE_COLOR, [{'if': False, 'exclude': [{'shape': 'circle'}, {'shape': 'square'}]}])
        self.assertEqual(result, [CIRCLE_RED, CIRCLE_BLUE, SQUARE_RED, SQUARE_BLUE])


class OrderingTest(unittest.TestCase):
    def test_extend_then_exclude_removes_the_extended_row(self):
        result = build_matrix({'shape': ['circle', 'square']}, [
            {'extend': {'shape': 'circle', 'sides': 0}},
            {'exclude': {'shape': 'circle'}},
        ])
        self.assertEqual(result, [{'shape': 'square'}])

    def test_exclude_then_extend_has_nothing_to_extend(self):
        result = build_matrix({'shape': ['circle', 'square']}, [
            {'exclude': {'shape': 'circle'}},
            {'extend': {'shape': 'circle', 'sides': 0}},
        ])
        self.assertEqual(result, [{'shape': 'square'}])

    def test_exclude_then_include_recreates_the_row(self):
        result = build_matrix({'shape': ['circle', 'square']}, [
            {'exclude': {'shape': 'circle'}},
            {'include': {'shape': 'circle', 'label': 'ring'}},
        ])
        self.assertEqual(result, [{'shape': 'square'}, {'shape': 'circle', 'label': 'ring'}])

    def test_rules_build_a_matrix_from_nothing(self):
        result = build_matrix({}, [
            {'include': {'shape': 'circle', 'sides': 0}},
            {'include': {'shape': 'square', 'sides': 4}},
            {'if': 'true', 'exclude': {'shape': 'square'}},
        ])
        self.assertEqual(result, [{'shape': 'circle', 'sides': 0}])


class EntryListBaseTest(unittest.TestCase):
    def test_entries_are_used_verbatim(self):
        entries = [{'animal': 'cat', 'sound': 'meow'}, {'animal': 'dog', 'sound': 'woof', 'trained': True}]
        self.assertEqual(build_matrix(entries, []), entries)

    def test_empty_list_gives_empty_matrix(self):
        self.assertEqual(build_matrix([], []), [])

    def test_axis_keys_are_the_union_of_entry_keys(self):
        result = build_matrix(
            [{'animal': 'cat'}, {'animal': 'dog', 'trained': True}],
            [{'include': {'trained': True, 'tricks': 3}}],
        )
        self.assertEqual(result, [{'animal': 'cat'}, {'animal': 'dog', 'trained': True, 'tricks': 3}])

    def test_include_with_a_new_axis_value_appends(self):
        result = build_matrix([{'animal': 'cat'}], [{'include': {'animal': 'parrot', 'sound': 'squawk'}}])
        self.assertEqual(result, [{'animal': 'cat'}, {'animal': 'parrot', 'sound': 'squawk'}])

    def test_inputs_are_not_mutated(self):
        entries = [{'animal': 'cat'}]
        axes = {'shape': ['circle']}
        build_matrix(entries, [{'extend': {'legs': 4}}])
        build_matrix(axes, [{'extend': {'legs': 4}}])
        self.assertEqual(entries, [{'animal': 'cat'}])
        self.assertEqual(axes, {'shape': ['circle']})


class ErrorTest(unittest.TestCase):
    def test_matrix_must_be_map_or_list(self):
        for bad in [None, 'circle', 3, True]:
            with self.subTest(bad=repr(bad)):
                with self.assertRaisesRegex(ValueError, "^'matrix' must be a map of axis names to value lists or a list of entries$"):
                    build_matrix(bad, [])

    def test_entry_list_items_must_be_objects(self):
        with self.assertRaisesRegex(ValueError, '^base matrix entry #1 must be an object$'):
            build_matrix([{'animal': 'cat'}, 'dog'], [])

    def test_rules_must_be_a_list(self):
        with self.assertRaisesRegex(ValueError, "^'rules' must be a list$"):
            build_matrix({}, {'exclude': {'shape': 'circle'}})

    def test_rule_must_be_an_object(self):
        with self.assertRaisesRegex(ValueError, '^rule #0 is not an object$'):
            build_matrix({}, ['exclude'])

    def test_rule_needs_exactly_one_kind(self):
        with self.assertRaisesRegex(ValueError, r"^rule #0 must have exactly one of 'include', 'extend', 'exclude' \(found: none\)$"):
            build_matrix({}, [{'if': True}])
        with self.assertRaisesRegex(ValueError, r"\(found: include, exclude\)$"):
            build_matrix({}, [{'include': {'shape': 'circle'}, 'exclude': {'shape': 'circle'}}])

    def test_skipped_rule_is_not_validated(self):
        self.assertEqual(build_matrix({'shape': ['circle']}, [{'if': False}]), [{'shape': 'circle'}])

    def test_body_must_be_object_or_list(self):
        with self.assertRaisesRegex(ValueError, "^rule #1 'extend' must be an object or a list of objects$"):
            build_matrix({}, [{'exclude': {}}, {'extend': 'sides'}])

    def test_list_body_items_must_be_objects(self):
        with self.assertRaisesRegex(ValueError, "^rule #0 'exclude' entry #1 must be an object$"):
            build_matrix({}, [{'exclude': [{'shape': 'circle'}, ['shape', 'square']]}])


class ScriptTest(unittest.TestCase):
    """matrix_builder.py as a command: base matrix JSON from '-' (stdin) or a
    file, rules JSON in --rules, result appended to GITHUB_OUTPUT."""

    def run_script(self, matrix_json, rules_json=None, matrix_file=None):
        with tempfile.TemporaryDirectory() as tmp:
            output_file = Path(tmp) / 'github_output'
            output_file.touch()
            if matrix_file is None:
                matrix_arg, stdin = '-', matrix_json
            else:
                matrix_arg, stdin = str(Path(tmp) / matrix_file), None
                Path(matrix_arg).write_text(matrix_json)
            command = [sys.executable, str(ACTION_DIR / 'matrix_builder.py')]
            if rules_json is not None:
                command.append(f'--rules={rules_json}')
            command.append(matrix_arg)
            proc = subprocess.run(
                command,
                input=stdin,
                capture_output=True,
                text=True,
                env={**os.environ, 'GITHUB_OUTPUT': str(output_file)},
            )
            outputs = read_outputs(output_file)
        return proc, outputs

    def test_matrix_from_stdin(self):
        proc, outputs = self.run_script('{"shape": ["circle", "square"]}', '[{"exclude": {"shape": "square"}}]')
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertEqual(outputs, {'matrix': '[{"shape":"circle"}]'})

    def test_matrix_from_file(self):
        proc, outputs = self.run_script(
            '[{"animal": "cat"}, {"animal": "dog"}]',
            '[{"extend": {"animal": "dog", "trained": true}}]',
            matrix_file='zoo.json',
        )
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertEqual(outputs, {'matrix': '[{"animal":"cat"},{"animal":"dog","trained":true}]'})

    def test_null_matrix_and_rules_default_to_empty(self):
        proc, outputs = self.run_script('null', 'null')
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertEqual(outputs, {'matrix': '[]'})

    def test_output_is_compact_json_and_log_is_grouped(self):
        proc, outputs = self.run_script('{"shape": ["circle", "square"], "color": ["red"]}', '[]')
        self.assertEqual(outputs['matrix'], '[{"shape":"circle","color":"red"},{"shape":"square","color":"red"}]')
        lines = proc.stdout.splitlines()
        self.assertEqual(lines[0], '::group::matrix-builder: 2 combination(s)')
        self.assertEqual(lines[-1], '::endgroup::')
        self.assertEqual(json.loads('\n'.join(lines[1:-1])), json.loads(outputs['matrix']))

    def test_engine_error_is_annotated_and_writes_no_output(self):
        proc, outputs = self.run_script('{"shape": ["circle"]}', '[{"include": {"shape": "circle"}, "exclude": {"shape": "circle"}}]')
        self.assertEqual(proc.returncode, 1)
        self.assertIn("::error::rule #0 must have exactly one of 'include', 'extend', 'exclude' (found: include, exclude)", proc.stdout)
        self.assertEqual(outputs, {})

    def test_rules_argument_is_required(self):
        proc, outputs = self.run_script('{}')
        self.assertEqual(proc.returncode, 2)
        self.assertIn('--rules', proc.stderr)
        self.assertEqual(outputs, {})


class ActionStepTest(unittest.TestCase):
    """The composite step's `run:` body, executed as the runner does
    (`bash -eo pipefail`) with the inputs in the variables action.yml maps
    them to, from a temporary directory standing in for the workspace."""

    @classmethod
    def setUpClass(cls):
        body = subprocess.run(
            ['yq', '.runs.steps[0].run', str(ACTION_DIR / 'action.yml')],
            capture_output=True, text=True, check=True,
        ).stdout
        cls.step = body.replace('${{ github.action_path }}', str(ACTION_DIR))

    def run_step(self, matrix='', matrix_file='', rules='[]', files=()):
        with tempfile.TemporaryDirectory() as workspace:
            for name, content in files:
                (Path(workspace) / name).write_text(content)
            output_file = Path(workspace) / 'github_output'
            output_file.touch()
            proc = subprocess.run(
                ['bash', '-eo', 'pipefail', '-c', self.step],
                capture_output=True,
                text=True,
                cwd=workspace,
                env={**os.environ, 'MATRIX': matrix, 'MATRIX_FILE': matrix_file, 'RULES': rules, 'GITHUB_OUTPUT': str(output_file)},
            )
            outputs = read_outputs(output_file)
        return proc, outputs

    def test_inline_yaml_inputs(self):
        proc, outputs = self.run_step(
            matrix='shape: [circle, square]\ncolor: [red, blue]\n',
            rules='- extend:\n    shape: circle\n    sides: 0\n- exclude:\n    shape: square\n    color: blue\n',
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(json.loads(outputs['matrix']), [
            {'shape': 'circle', 'color': 'red', 'sides': 0},
            {'shape': 'circle', 'color': 'blue', 'sides': 0},
            {'shape': 'square', 'color': 'red'},
        ])

    def test_matrix_file_is_resolved_against_the_workspace(self):
        zoo = '- animal: cat\n  sound: meow\n- animal: dog\n  sound: woof\n'
        proc, outputs = self.run_step(matrix_file='zoo.yml', files=[('zoo.yml', zoo)])
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(json.loads(outputs['matrix']), [
            {'animal': 'cat', 'sound': 'meow'},
            {'animal': 'dog', 'sound': 'woof'},
        ])

    def test_matrix_and_matrix_file_are_mutually_exclusive(self):
        proc, outputs = self.run_step(matrix='shape: [circle]\n', matrix_file='zoo.yml', files=[('zoo.yml', '[]\n')])
        self.assertEqual(proc.returncode, 1)
        self.assertIn("::error::'matrix' and 'matrix-file' are mutually exclusive", proc.stdout)
        self.assertEqual(outputs, {})

    def test_empty_matrix_is_built_from_rules(self):
        proc, outputs = self.run_step(rules='- include:\n    shape: circle\n- include:\n    shape: square\n')
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(outputs, {'matrix': '[{"shape":"circle"},{"shape":"square"}]'})

    def test_yaml_11_booleans_stay_strings(self):
        proc, outputs = self.run_step(matrix='flag: [yes, no]\n', rules='- if: off\n  exclude:\n    flag: yes\n')
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(outputs, {'matrix': '[{"flag":"yes"},{"flag":"no"}]'})

    def test_expanded_expression_strings_guard_rules(self):
        proc, outputs = self.run_step(
            matrix='shape: [circle, square]\n',
            rules='- if: true\n  exclude:\n    shape: square\n- if: "false"\n  exclude:\n    shape: circle\n',
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(outputs, {'matrix': '[{"shape":"circle"}]'})

    def test_invalid_yaml_fails_the_step(self):
        proc, outputs = self.run_step(matrix='shape: [circle]\n', rules='exclude: shape: circle\n')
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('mapping values are not allowed', proc.stderr)
        self.assertEqual(outputs, {})

    def test_missing_matrix_file_fails_the_step(self):
        proc, outputs = self.run_step(matrix_file='missing.yml')
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('missing.yml', proc.stderr)
        self.assertEqual(outputs, {})


if __name__ == '__main__':
    unittest.main()
