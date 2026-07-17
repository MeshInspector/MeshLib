#!/usr/bin/env python3
r"""Inject the MeshLib C++ doc comments into the emscripten-generated ``bindings.d.mts``.

The wasm build's ``--emit-tsd`` declaration file carries no documentation. This tool enriches it in
place, reusing the C++ API comments that the embind bindings already wrap, so IDEs (on hover) and
the Doxygen ``Js`` module read the same ``/** */`` blocks. Two inputs are combined:

* ``--comments`` : the Doxygen XML dir produced by ``scripts/npm/Doxyfile`` (run that first). Holds
  the C++ brief/detailed descriptions, keyed by qualified name (e.g. ``MR::BooleanResult::valid``).
* ``--bindings`` : the hand-written embind sources (``source/MRWasmModule/*.cpp``), mapping each JS
  name to its C++ entity; bindings that name no C++ target fall back to matching by name.

Usage: ``doxygen scripts/npm/Doxyfile && inject_tsdoc.py --bindings source/MRWasmModule bindings.d.mts``
(rewrites the file in place; ``--output -`` writes to stdout).
"""
import argparse
import glob
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Must match XML_OUTPUT in scripts/npm/Doxyfile (both are run from the repo root).
DEFAULT_XML_DIR = "build/wasm_cpp_comments"


# --------------------------------------------------------------------------- comment DB (Doxygen XML)

# <simplesect kind=...> / <parameterlist kind=...> -> the JSDoc/Doxygen command to re-emit.
_SIMPLESECT = {"return": "@returns", "note": "@note", "warning": "@warning", "see": "@see",
               "remark": "@remark", "attention": "@attention", "since": "@since"}
_PARAMLIST = {"param": "@param", "templateparam": "@tparam", "exception": "@throws", "retval": "@retval"}


def _inline(elem):
    """Render an element's mixed text/child content to a string (ElementTree text+tail model)."""
    parts = [elem.text or ""]
    for child in elem:
        parts.append(_render(child))
        parts.append(child.tail or "")
    return "".join(parts)


def _render(node):
    """Render one Doxygen-XML description node to comment text, preserving @param/@note etc."""
    tag = node.tag
    if tag == "computeroutput":
        return f"`{_inline(node).strip()}`"
    if tag == "para":
        return _inline(node).strip() + "\n\n"
    if tag == "parameterlist":
        cmd = _PARAMLIST.get(node.get("kind"), "@param")
        lines = []
        for item in node.findall("parameteritem"):
            names = " ".join(_inline(n).strip() for n in item.findall(".//parametername"))
            desc = " ".join(_inline(p).strip() for p in item.findall(".//parameterdescription/para"))
            lines.append(f"{cmd} {names} {desc}".rstrip())
        return "\n".join(lines) + "\n"
    if tag == "simplesect":
        cmd = _SIMPLESECT.get(node.get("kind"), "")
        body = " ".join(_inline(p).strip() for p in node.findall("para")).strip()
        return (f"{cmd} {body}".strip() if cmd else body) + "\n"
    if tag == "itemizedlist":
        return "\n" + "\n".join(f"- {_inline(li).strip()}" for li in node.findall("listitem")) + "\n"
    if tag == "orderedlist":
        items = node.findall("listitem")
        return "\n" + "\n".join(f"{i + 1}. {_inline(li).strip()}" for i, li in enumerate(items)) + "\n"
    if tag in ("programlisting", "image", "xrefsect", "table", "formula"):
        return ""  # C++ \snippet/\code, images, xref dumps: irrelevant or noisy in JS docs
    return _inline(node)  # bold / emphasis / ref / ulink / heading: keep the inner text


def _describe(elem):
    """Brief + detailed description of a compounddef/memberdef as comment text, or None if empty.

    The brief line is prefixed with ``@brief``."""
    brief = elem.find("briefdescription")
    detail = elem.find("detaileddescription")
    blocks = []
    if brief is not None and _inline(brief).strip():
        blocks.append("@brief " + _inline(brief).strip())
    if detail is not None and _inline(detail).strip():
        blocks.append(_inline(detail).strip())
    out = []
    for ln in "\n\n".join(blocks).splitlines():   # collapse runs of blank lines to one
        ln = ln.rstrip()
        if not ln and (not out or not out[-1]):
            continue
        out.append(ln)
    while out and not out[-1]:
        out.pop()
    return "\n".join(out) or None


def load_comment_db(xml_dir):
    """Parse a Doxygen XML dir into three lookup tables:

    - db:        {qualified_name: comment_text}
    - by_tail:   {tail_key: [(qualified_name, comment_text)]}
    - params_db: {qualified_name: [[param_name, ...]]}  (one inner list per overload)
    """
    db = {}
    by_tail = {}
    params_db = {}

    def add(qual, text):
        if not qual or not text or qual in db:
            return
        db[qual] = text
        parts = qual.split("::")
        by_tail.setdefault(parts[-1], []).append((qual, text))
        if len(parts) >= 2:
            by_tail.setdefault("::".join(parts[-2:]), []).append((qual, text))

    for path in sorted(glob.glob(os.path.join(xml_dir, "*.xml"))):
        if os.path.basename(path) in ("index.xml", "Doxyfile.xml"):
            continue
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        for cd in root.iter("compounddef"):
            cname = cd.findtext("compoundname")
            if cd.get("kind") in ("class", "struct", "namespace", "union"):
                add(cname, _describe(cd))
            for md in cd.iter("memberdef"):
                qual = md.findtext("qualifiedname") or (f"{cname}::{md.findtext('name')}" if cname else None)
                add(qual, _describe(md))
                if md.get("kind") == "function" and qual:
                    params_db.setdefault(qual, []).append(
                        [(p.findtext("declname") or "").strip() for p in md.findall("param")])
                if md.get("kind") == "enum" and qual:
                    for ev in md.findall("enumvalue"):
                        add(f"{qual}::{ev.findtext('name')}", _describe(ev))
    return db, by_tail, params_db


# --------------------------------------------------------------------------- embind map

_MR = "MR::"

# Synthetic embind converters have no 1:1 C++ declaration; name their sole param descriptively.
_SYNTH_PARAMS = {"fromArray": ["array"], "fromIndices": ["indices"]}


def _split_params(s):
    """Split a C++ parameter list on top-level commas (respecting <> () [] {} nesting)."""
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "(<[{":
            depth += 1
        elif ch in ")>]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return parts


def _param_name(param):
    """The declared name of one C++ parameter (its trailing identifier), or None."""
    ids = re.findall(r"[A-Za-z_]\w*", param)
    return ids[-1] if ids else None


def _lambda_params(text):
    """Parameter names of the first ``+[](...)`` embind wrapper in `text` (receiver included).

    `text` may span several joined lines, since a wrapper's parameter list can spill past the line
    its binding name is on. None if there is no wrapper or its parentheses never balance."""
    m = re.search(r"\+\s*\[\s*\]\s*\(", text)
    if not m:
        return None
    depth, i = 1, m.end()
    while i < len(text) and depth:
        c = text[i]
        if c in "(<[{":
            depth += 1
        elif c in ")>]}":
            depth -= 1
        i += 1
    if depth:
        return None
    return [_param_name(p) for p in _split_params(text[m.end():i - 1])]


def _qualify(cpp_type):
    """Normalize an embind C++ type spelling to a fully-qualified name for DB lookup."""
    t = re.sub(r"<.*>", "", cpp_type).strip().lstrip("&")
    if not t or t.startswith(("std::", _MR)):
        return t
    return _MR + t


class EmbindMap:
    def __init__(self):
        self.classes = {}        # js class/enum -> cpp qualified type
        self.members = {}        # (js class, js member) -> cpp target or None
        self.frees = {}          # js free function -> cpp target or None
        self.enum_values = {}    # (js enum, js value) -> cpp enumerator or None
        self.free_params = {}    # js free function -> list of [param names] (one per overload)
        self.member_params = {}  # (js class, js member) -> list of [param names]; receiver dropped
        self.ctor_params = {}    # js class -> list of [param names] from a factory .constructor(+[]...)


def load_embind_map(sources_dir):
    m = EmbindMap()
    reg = re.compile(r'emscripten::(class_|enum_)<\s*(.+?)\s*>\(\s*"([\w]+)"')
    member = re.compile(r'\.(function|property|class_function)\(\s*"([\w]+)"\s*,\s*(&[\w:]+)?')
    free = re.compile(r'emscripten::function\(\s*"([\w]+)"\s*,\s*(&[\w:]+)?')
    ctor = re.compile(r'\.constructor\(\s*\+\s*\[\s*\]')  # a factory constructor's +[] wrapper
    value = re.compile(r'\.value\(\s*"([\w]+)"\s*,\s*([\w:]+)')

    for cpp in sorted(Path(sources_dir).glob("*.cpp")):
        current = None  # (js name, kind)
        lines = cpp.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines):
            window = "\n".join(lines[i:i + 8])  # a wrapper's +[](...) list may spill onto later lines
            mr = reg.search(line)
            if mr:
                kind, cpp_type, js = mr.group(1), mr.group(2), mr.group(3)
                m.classes[js] = _qualify(cpp_type)
                current = (js, kind)
            fm = free.search(line)
            if fm:
                js, target = fm.group(1), fm.group(2)
                m.frees[js] = target[1:] if target else None
                if not target:  # +[]{} wrapper: capture its param names (frees have no receiver)
                    params = _lambda_params(window)
                    if params and all(params):
                        m.free_params.setdefault(js, []).append(params)
            mm = member.search(line)
            if mm and current:
                mkind, js, target = mm.group(1), mm.group(2), mm.group(3)
                if current[1] == "enum_":
                    continue
                m.members[(current[0], js)] = target[1:] if target else None
                if not target and mkind != "property":  # a +[]{} wrapper method
                    params = _lambda_params(window)
                    if params and mkind == "function":
                        params = params[1:]  # drop the receiver (*this) of an instance method
                    if params and all(params):
                        m.member_params.setdefault((current[0], js), []).append(params)
            if ctor.search(line) and current and current[1] == "class_":
                params = _lambda_params(window)  # the factory lambda's params are the JS ctor's
                if params and all(params):
                    m.ctor_params.setdefault(current[0], []).append(params)
            vm = value.search(line)
            if vm and current and current[1] == "enum_":
                m.enum_values[(current[0], vm.group(1))] = vm.group(2)
    return m


# --------------------------------------------------------------------------- comment format

_DROP_CMD = re.compile(r"^\s*[\\@](ingroup|defgroup|addtogroup|relatesalso|relates|snippet|\{|\})\b.*$")


def normalize(text, indent):
    """Wrap a reconstructed comment body into an indented JSDoc/TSDoc /** */ block."""
    lines = []
    for raw in text.splitlines():
        ln = raw.rstrip()
        ln = re.sub(r"^\s*\*? ?", "", ln)         # strip any residual leading ` * `
        if _DROP_CMD.match(ln):
            continue
        ln = re.sub(r"(^|\s)\\(\w+)", r"\1@\2", ln)  # Doxygen \cmd -> @cmd (tsc-friendly)
        lines.append(ln)
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return None
    out = [f"{indent}/**"]
    out += [f"{indent} * {ln}".rstrip() for ln in lines]
    out.append(f"{indent} */")
    return "\n".join(out)


# --------------------------------------------------------------------------- .d.mts injection

class Injector:
    def __init__(self, db, by_tail, emb, params_db):
        self.db, self.by_tail, self.emb, self.params_db = db, by_tail, emb, params_db
        self.hits = 0
        self.misses = []

    def _lookup(self, *candidates):
        for c in candidates:
            if c and c in self.db:
                return self.db[c]
        # unique tail match (e.g. "Mesh::volume") as a last resort
        for c in candidates:
            if not c:
                continue
            tail = "::".join(c.split("::")[-2:]) if "::" in c else c
            hit = self.by_tail.get(tail)
            if hit and len({t for _, t in hit}) == 1:
                return hit[0][1]
        return None

    def comment_for_type(self, js):
        cpp = self.emb.classes.get(js)
        return self._lookup(cpp, f"{_MR}{js}"), (cpp or f"{_MR}{js}")

    def comment_for_member(self, js_cls, js_mem):
        cpp_cls = self.emb.classes.get(js_cls, f"{_MR}{js_cls}")
        target = self.emb.members.get((js_cls, js_mem))
        return self._lookup(target, f"{cpp_cls}::{js_mem}", f"{_MR}{js_cls}::{js_mem}"), f"{cpp_cls}::{js_mem}"

    def comment_for_free(self, js):
        target = self.emb.frees.get(js)
        return self._lookup(target, f"{_MR}{js}"), (target or f"{_MR}{js}")

    def comment_for_value(self, js_enum, js_val):
        cpp_enum = self.emb.classes.get(js_enum, f"{_MR}{js_enum}")
        target = self.emb.enum_values.get((js_enum, js_val))
        return self._lookup(target, f"{cpp_enum}::{js_val}"), f"{cpp_enum}::{js_val}"

    # -- real parameter names (to replace emit-tsd's positional _0/_1/...) --------------------
    # Prefer the wrapper's names (the JS signature), else the arity-matched C++ declnames.

    def _cpp_params(self, candidates, arity):
        for c in candidates:
            if not c:
                continue
            for names in self.params_db.get(c, []):
                if len(names) == arity and all(names):
                    return names
        return None

    def param_names_for_free(self, js, arity):
        for names in self.emb.free_params.get(js, []):
            if len(names) == arity and all(names):
                return names
        return self._cpp_params([self.emb.frees.get(js), f"{_MR}{js}"], arity)

    def param_names_for_member(self, js_cls, js_mem, arity):
        for names in self.emb.member_params.get((js_cls, js_mem), []):
            if len(names) == arity and all(names):
                return names
        cpp_cls = self.emb.classes.get(js_cls, f"{_MR}{js_cls}")
        cpp = self._cpp_params(
            [self.emb.members.get((js_cls, js_mem)), f"{cpp_cls}::{js_mem}", f"{_MR}{js_cls}::{js_mem}"], arity)
        if cpp:
            return cpp
        synth = _SYNTH_PARAMS.get(js_mem)  # synthetic converters have no C++ decl to source from
        return synth if synth and len(synth) == arity else None

    def param_names_for_ctor(self, js_cls, arity):
        for names in self.emb.ctor_params.get(js_cls, []):  # a factory .constructor(+[]...) wrapper
            if len(names) == arity and all(names):
                return names
        cpp_cls = self.emb.classes.get(js_cls, f"{_MR}{js_cls}")  # else the C++ ctor's declnames
        return self._cpp_params([f"{cpp_cls}::{cpp_cls.split('::')[-1]}", f"{_MR}{js_cls}::{js_cls}"], arity)

    @staticmethod
    def _rename(s, names):
        """Replace emit-tsd's positional ``_0``..``_N`` identifiers in `s` with `names`."""
        return re.sub(r"\b_(\d+)\b",
                      lambda mo: names[int(mo.group(1))] if int(mo.group(1)) < len(names) else mo.group(0), s)

    def run(self, src):
        lines = src.splitlines()
        out = []
        ctx = None        # current interface/class name, or ("module",) inside EmbindModule
        stat_cls = None   # current static-block class name inside EmbindModule
        for line in lines:
            s = line.strip()
            indent = line[: len(line) - len(line.lstrip())]

            # already-documented line: leave as is
            prev_doc = out and out[-1].strip().endswith("*/")

            comment, key, names = None, None, None
            iface = re.match(r"(?:export )?interface (\w+)", s)
            type_alias = re.match(r"(?:export )?type (\w+)\s*=", s)
            if iface:
                name = iface.group(1)
                if name == "EmbindModule":
                    ctx, stat_cls = "module", None
                elif name in ("WasmModule", "ClassHandle"):
                    ctx = None
                else:
                    ctx = name
                    comment, key = self.comment_for_type(name)
            elif type_alias and ctx is None:
                # value_object / register_type aliases (`type PointOnFace = {...}`): document from the
                # same-named C++ struct via the naming convention; skip silently if none exists.
                name = type_alias.group(1)
                comment, _ = self.comment_for_type(name)
                if s.endswith("{"):  # multiline object alias: document its fields too (as members below)
                    ctx = name
            elif ctx == "module":
                if re.match(r"(\w+):\s*\{.+Value<\d+>", s):        # enum object literal
                    comment, key = self.comment_for_type(re.match(r"(\w+):", s).group(1))
                elif re.match(r"(\w+):\s*\{$", s):                  # open a class's static block
                    stat_cls = re.match(r"(\w+):", s).group(1)
                elif s in ("};", "}"):                              # close static block / the module
                    if stat_cls:
                        stat_cls = None
                    else:
                        ctx = None
                elif stat_cls and s.startswith("new(") and s.endswith(";"):
                    arity = len(re.findall(r"\b_\d+\b", s))  # constructor: names only, no comment
                    names = self.param_names_for_ctor(stat_cls, arity) if arity else None
                elif re.match(r"(\w+)\(.*\):", s) and not s.startswith("new("):
                    fn = re.match(r"(\w+)\(", s).group(1)
                    arity = len(re.findall(r"\b_\d+\b", s))
                    if stat_cls:
                        comment, key = self.comment_for_member(stat_cls, fn)
                        names = self.param_names_for_member(stat_cls, fn, arity) if arity else None
                    else:
                        comment, key = self.comment_for_free(fn)
                        names = self.param_names_for_free(fn, arity) if arity else None
            elif ctx and ctx != "module":
                g = re.match(r"get (\w+)\(\):", s)
                mth = re.match(r"(\w+)\((.*)\):", s)
                prop = re.match(r"(?:readonly )?(\w+)\??:", s)
                nm = g or mth or prop
                if nm and s.endswith(";"):
                    comment, key = self.comment_for_member(ctx, nm.group(1))
                    if mth and not g:
                        arity = len(re.findall(r"\b_\d+\b", s))
                        names = self.param_names_for_member(ctx, mth.group(1), arity) if arity else None
                if s in ("}", "};"):
                    ctx = None

            if comment and not prev_doc:
                out.append(normalize(comment, indent) or "")
                self.hits += 1
            elif key and not prev_doc and re.search(r"[\w>]\s*\(|:\s|\{$", s) and s not in ("};", "}"):
                self.misses.append(key)
            if names:
                renamed = self._rename(s, names)
                if renamed != s:
                    line = indent + renamed
            out.append(line)
        return "\n".join(out) + ("\n" if src.endswith("\n") else "")


# --------------------------------------------------------------------------- consistency report

def report_param_mismatches(emb, params_db):
    """Print each ``+[]{}`` wrapper whose parameter names differ from the matched C++ declaration.

    Direct ``&`` binds are not checked."""
    def cpp_for(candidates, arity):
        for c in candidates:
            if not c:
                continue
            for names in params_db.get(c, []):
                if len(names) == arity and all(names):
                    return names
        return None

    diverged = unresolved = 0

    def emit(label, w, candidates):
        nonlocal diverged, unresolved
        cpp = cpp_for(candidates, len(w))
        if not cpp:
            unresolved += 1
            return
        diff = [(a, b) for a, b in zip(w, cpp) if a != b]
        if diff:
            diverged += 1
            print(f"{label}({', '.join(w)})  ->  ({', '.join(cpp)})")
            for a, b in diff:
                print(f"    {a}  ->  {b}")

    for js, overloads in sorted(emb.free_params.items()):
        for w in overloads:
            emit(js, w, [emb.frees.get(js), f"{_MR}{js}"])
    for (cls, mem), overloads in sorted(emb.member_params.items()):
        cpp_cls = emb.classes.get(cls, f"{_MR}{cls}")
        for w in overloads:
            emit(f"{cls}.{mem}", w, [emb.members.get((cls, mem)), f"{cpp_cls}::{mem}", f"{_MR}{cls}::{mem}"])

    print(f"\n[report] {diverged} wrapper(s) diverge from C++; "
          f"{unresolved} unresolved (no arity-matched C++ decl)")


def main():
    ap = argparse.ArgumentParser(description="Document bindings.d.mts with the MeshLib C++ doc comments")
    ap.add_argument("tsd", nargs="?", help="the emscripten --emit-tsd .d.mts to enrich (in place)")
    ap.add_argument("--bindings", required=True, help="dir with embind source/MRWasmModule/*.cpp")
    ap.add_argument("--comments", default=DEFAULT_XML_DIR,
                    help=f"Doxygen XML dir from scripts/npm/Doxyfile (default: {DEFAULT_XML_DIR})")
    ap.add_argument("--output", default=None, help="output path, or '-' for stdout (default: in place)")
    ap.add_argument("--report-params", action="store_true",
                    help="list wrapper params that diverge from the C++ decl, then exit")
    args = ap.parse_args()

    db, by_tail, params_db = load_comment_db(args.comments)
    emb = load_embind_map(args.bindings)

    if args.report_params:
        report_param_mismatches(emb, params_db)
        return
    if not args.tsd:
        ap.error("the 'tsd' argument is required unless --report-params is given")

    inj = Injector(db, by_tail, emb, params_db)
    result = inj.run(Path(args.tsd).read_text(encoding="utf-8"))

    if args.output == "-":
        sys.stdout.write(result)
    else:
        Path(args.output or args.tsd).write_text(result, encoding="utf-8")

    uniq_miss = sorted(set(inj.misses))
    sys.stderr.write(f"[inject_tsdoc] documented {inj.hits} symbols from {len(db)} C++ comments; "
                     f"{len(uniq_miss)} undocumented\n")
    for k in uniq_miss[:40]:
        sys.stderr.write(f"  undocumented: {k}\n")


if __name__ == "__main__":
    main()
