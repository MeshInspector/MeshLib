# matrix-builder

Declarative, ordered include / extend / exclude rules for building GitHub
Actions matrices.

Unlike GitHub's native `matrix.include` / `matrix.exclude` — where `exclude`
runs first and `include` is then applied unconditionally, so a later
`include` can resurrect entries you just excluded — every rule here is
evaluated in source order. The final matrix is exactly what the YAML reads
top-to-bottom.

## Inputs

| Name          | Required | Default | Description                                                                              |
|---------------|----------|---------|------------------------------------------------------------------------------------------|
| `matrix`      | no       | `{}`    | YAML/JSON base matrix: axis map (cartesian product) or list of entries (used as-is). Mutually exclusive with `matrix-file`. |
| `matrix-file` | no       | —       | Path of a YAML/JSON file to load the base matrix from (same forms as `matrix`). Mutually exclusive with `matrix`. |
| `rules`       | no       | `[]`    | YAML/JSON list of rules, applied in order.                                               |

All inputs accept YAML or JSON (YAML is a superset). YAML 1.1 booleans
(`yes` / `no` / `on` / `off`) are intentionally not coerced — they remain
strings, matching what you see in your workflow. `matrix-file` paths are
resolved against the workspace root, so the file must be checked out (a
sparse checkout of its directory is enough).

## Outputs

| Name     | Description                                                                       |
|----------|-----------------------------------------------------------------------------------|
| `matrix` | JSON-encoded list of resulting combinations. Consume via `fromJSON(...)`.         |

## Grammar

### Base matrix

The base matrix comes from the `matrix` input, or — when `matrix-file` is
set instead — from a YAML/JSON file in the repository. Either way it takes
one of two forms.

**Axis map** — a map from axis name to a list of values. Scalar values are
treated as one-element lists. Missing or `{}` starts from an empty list of
entries.

```yaml
matrix:
  platform:
    - x86_64
    - aarch64
  config:
    - Debug
    - Release
```

produces 4 base entries:

```json
[
  {"platform": "x86_64",  "config": "Debug"},
  {"platform": "x86_64",  "config": "Release"},
  {"platform": "aarch64", "config": "Debug"},
  {"platform": "aarch64", "config": "Release"}
]
```

**Entry list** — a list of include-style entries, taken as the base entries
verbatim (no cartesian product). The axis keys — what `include` / `extend`
rules match on — are the union of the entries' keys.

```yaml
matrix:
  - { distro: ubuntu22, arch: x64 }
  - { distro: ubuntu24, arch: arm64 }
```

### Loading the base matrix from a file (`matrix-file`)

`matrix-file` names a YAML or JSON file (workspace-relative) holding the
base matrix in either form above, so an inventory file can be shared
between workflows and shell scripts:

```yaml
# .github/workflows/matrix/docker-images-linux.json:
#   [ {"distro": "ubuntu22", "arch": "x64"}, ... ]
- uses: ./.github/actions/matrix-builder
  with:
    matrix-file: .github/workflows/matrix/docker-images-linux.json
    rules: |
      - if: ${{ inputs.disable_ubuntu_x64 }}
        exclude:
          - distro: ubuntu22
            arch: x64
          - distro: ubuntu24
            arch: x64
```

`matrix` and `matrix-file` are mutually exclusive.

### Rules

Each rule must contain exactly one of `include`, `extend`, `exclude`, and
may optionally be guarded by `if:`.

The keyword's value may be **either a single object or a list of objects**.
The list form groups several related operations under one keyword (and
under one `if`); the operations are applied in source order. The two forms
below are equivalent:

```yaml
# Single object form:
- extend:
    py-version: "3.11"
    os: "fedora:37"
- extend:
    py-version: "3.12"
    os: "fedora:39"

# Grouped list form (shorter when you have several related operations):
- extend:
    - py-version: "3.11"
      os: "fedora:37"
    - py-version: "3.12"
      os: "fedora:39"
```

The `if:` value is GitHub-expanded *before* the action runs (i.e. by the
time the action sees `rules`, any `${{ ... }}` has been resolved to a
literal string). Truthiness:

  - YAML `true` → truthy; `false` / `null` / missing → falsy.
  - Strings: trim + lowercase; `""`, `"false"`, `"0"`, `"no"`, `"off"` →
    falsy; anything else → truthy.
  - Numbers: zero is falsy, anything else truthy.

Rules with a falsy `if:` are skipped entirely.

#### `include` — extend matching rows, or add a new one

Mirrors GitHub's native `matrix.include`:

  1. The rule's keyset is split into *axis keys* (those that exist in the
     base `matrix`) and *extra keys*.
  2. For every existing entry whose values satisfy all the axis-key
     criteria, the extra keys are merged into that entry.
  3. If no entry matched and the rule has at least one axis key, the rule
     is appended as a new entry containing exactly its own keys.
  4. If the rule has no axis keys at all, it's appended as a new entry
     unconditionally.

#### `extend` — extend matching rows only (never adds)

Strict variant of `include`:

  1. Same axis-key / extra-key split.
  2. For every existing entry matching the axis-key criteria, merge in the
     extra keys.
  3. If no entry matched, the rule is a silent no-op. **`extend` never
     appends new entries.**
  4. With no axis keys, every entry trivially matches, so the extra keys
     are merged into all entries — useful for "set this field on
     everything".

Use `extend` when you want to attach attributes to existing rows but never
accidentally introduce a new combination.

#### `exclude` — drop matching rows

Removes every entry where, for each key in the rule, the entry has that
key with the matching value. Keys not present in an entry don't match
(same as GitHub's native `matrix.exclude`).

## Examples

### 1. Simple disable-flag filtering

```yaml
- uses: ./.github/actions/matrix-builder
  id: m
  with:
    matrix: |
      platform:
        - x86_64
        - aarch64
    rules: |
      - extend:
          platform: x86_64
          runner: ubuntu-latest
      - extend:
          platform: aarch64
          runner: ubuntu-24.04-arm
      - if: ${{ inputs.disable_x64 }}
        exclude:
          platform: x86_64
```

With `disable_x64 = false`:

```json
[
  {"platform": "x86_64",  "runner": "ubuntu-latest"},
  {"platform": "aarch64", "runner": "ubuntu-24.04-arm"}
]
```

With `disable_x64 = true`:

```json
[{"platform": "aarch64", "runner": "ubuntu-24.04-arm"}]
```

### 2. Cross-product with per-axis attributes

Declare both axes; group the per-axis attributes under list-form `extend`
rules.

```yaml
- uses: ./.github/actions/matrix-builder
  id: m
  with:
    matrix: |
      platform:
        - x86_64
        - aarch64
      py-version:
        - "3.11"
        - "3.12"
    rules: |
      - extend:
          - platform: x86_64
            runner: ubuntu-latest
          - platform: aarch64
            runner: ubuntu-24.04-arm
      - extend:
          - py-version: "3.11"
            os: "fedora:37"
          - py-version: "3.12"
            os: "fedora:39"
```

Produces 4 fully-fleshed entries — every combination of platform and
py-version, with both the platform's runner and the py-version's os.

### 3. Order-dependent rewrites (the whole reason this action exists)

```yaml
rules: |
  - extend:
      platform: x86_64
      flavor: stable
  - exclude:
      platform: x86_64
      flavor: stable
  - include:
      platform: x86_64
      flavor: nightly
```

  - After step 1: `x86_64` row has `flavor: stable`.
  - After step 2: the `x86_64` row is gone.
  - After step 3: a fresh `x86_64` row is added with `flavor: nightly`.

With GitHub's native include/exclude this is impossible — `include` is
applied unconditionally after `exclude`, so the original `x86_64` row
would survive.

### 4. Building a matrix entirely from rules

```yaml
- uses: ./.github/actions/matrix-builder
  with:
    matrix: '{}'
    rules: |
      - include:
          platform: x86_64
          runner: ubuntu-latest
      - include:
          platform: aarch64
          runner: ubuntu-24.04-arm
```

Empty base + `include` rules with no axis-key conflicts → each `include`
appends a fresh entry.

### 5. Apply a default to every row

```yaml
- uses: ./.github/actions/matrix-builder
  with:
    matrix: |
      platform: [x86_64, aarch64]
    rules: |
      - extend:
          compiler: clang
```

`extend` with no axis keys attaches `compiler: clang` to every entry.

## Layout

This is a composite action. Its single step converts each input to JSON
with `yq -o json .` and runs `matrix_builder.py` with the base matrix on
stdin and the rules in `--rules`; the script appends the `matrix` output to
`GITHUB_OUTPUT`. Both `yq` and `python3` come from the runner's PATH.

  - `action.yml` — action metadata (inputs, outputs) and the step above.
  - `matrix_builder.py` — the rules engine and the script entry point. Read
    this for the canonical semantics.
  - `tests/test_matrix_builder.py` — `unittest` suite for the engine, the
    script's command line, and the step body run through `bash` and `yq`.
    Run from this directory: `python3 -m unittest discover -s tests`.
