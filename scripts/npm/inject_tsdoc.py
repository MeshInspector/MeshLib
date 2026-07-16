#!/usr/bin/env python3
r"""Inject the MeshLib C++ doc comments into the emscripten-generated ``bindings.d.mts``.

The wasm build emits a *structural* TypeScript declaration file (``--emit-tsd``) that carries
no documentation, because embind has no doc-comment mechanism. This tool enriches it in place
so the one file documents the JS API for both consumers: IDEs read the ``/** */`` blocks on
hover, and the Doxygen ``Js`` module (via ``preprocessors/typescript.py``) renders the same
blocks.

The documentation is *reused from C++* — the embind bindings wrap already-documented MeshLib
entities. Run ``doxygen scripts/npm/Doxyfile`` first to extract the C++ API comments as XML, then
this tool injects them. Two sources are combined:

* ``--comments`` : the Doxygen XML directory produced by ``scripts/npm/Doxyfile`` (defaults to its
  static ``build/wasm_cpp_comments`` output; may also point at the docs pipeline's ``xml_Cpp``). Each
  ``<compounddef>`` (class/struct/namespace) and ``<memberdef>`` (function/method/field/enum) is
  keyed by its ``<qualifiedname>`` (e.g. ``MR::BooleanResult::valid``); its ``briefdescription`` +
  ``detaileddescription`` are rebuilt into a clean comment (brief re-marked ``@brief``, ``\param`` ->
  ``@param``, ``\note`` -> ``@note``, ``<computeroutput>`` -> backticks; C++ ``\snippet`` code dropped).
* ``--bindings`` : the hand-written embind sources (``source/MRWasmModule/*.cpp``), scanned for
  the JS-name -> C++-entity map (``class_<T>("Js")``, ``.function("js", &Qual::m)``,
  ``enum_<E>(...).value(...)``). ``+[]{}`` lambda wrappers expose no C++ target, so those fall
  back to the 1:1 naming convention (the JS name mirrors the C++ name) to key the comment DB.

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

# Static Doxygen XML location, matching XML_OUTPUT in scripts/npm/Doxyfile (relative to the repo
# root, where both doxygen and this script are run). Keep the two in sync.
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
    """Rebuild brief + detailed descriptions of a compounddef/memberdef into clean comment text.

    The brief is re-marked with ``@brief`` so the split survives Doxygen re-parsing the injected
    comment in the Js module (matching how the Cpp/Py modules show one-line summaries)."""
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
    """Doxygen XML dir -> ({qualified_name: comment_text}, {tail_key: [(qual, text)]})."""
    db = {}
    by_tail = {}

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
                if md.get("kind") == "enum" and qual:
                    for ev in md.findall("enumvalue"):
                        add(f"{qual}::{ev.findtext('name')}", _describe(ev))
    return db, by_tail


# --------------------------------------------------------------------------- embind map

_MR = "MR::"


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


def load_embind_map(sources_dir):
    m = EmbindMap()
    reg = re.compile(r'emscripten::(class_|enum_)<\s*(.+?)\s*>\(\s*"([\w]+)"')
    member = re.compile(r'\.(?:function|property|class_function)\(\s*"([\w]+)"\s*,\s*(&[\w:]+)?')
    free = re.compile(r'emscripten::function\(\s*"([\w]+)"\s*,\s*(&[\w:]+)?')
    value = re.compile(r'\.value\(\s*"([\w]+)"\s*,\s*([\w:]+)')

    for cpp in sorted(Path(sources_dir).glob("*.cpp")):
        current = None  # (js name, kind)
        for line in cpp.read_text(encoding="utf-8", errors="replace").splitlines():
            mr = reg.search(line)
            if mr:
                kind, cpp_type, js = mr.group(1), mr.group(2), mr.group(3)
                m.classes[js] = _qualify(cpp_type)
                current = (js, kind)
            fm = free.search(line)
            if fm:
                m.frees[fm.group(1)] = fm.group(2)[1:] if fm.group(2) else None
                # a top-level function() ends any pending class chain
                if not line.lstrip().startswith("."):
                    current = current if fm.group(1) in m.classes else current
            mm = member.search(line)
            if mm and current:
                js, target = mm.group(1), mm.group(2)
                if current[1] == "enum_":
                    continue
                m.members[(current[0], js)] = target[1:] if target else None
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
    def __init__(self, db, by_tail, emb):
        self.db, self.by_tail, self.emb = db, by_tail, emb
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

            comment, key = None, None
            iface = re.match(r"(?:export )?interface (\w+)", s)
            if iface:
                name = iface.group(1)
                if name == "EmbindModule":
                    ctx, stat_cls = "module", None
                elif name in ("WasmModule", "ClassHandle"):
                    ctx = None
                else:
                    ctx = name
                    comment, key = self.comment_for_type(name)
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
                elif re.match(r"(\w+)\(.*\):", s) and not s.startswith("new("):
                    fn = re.match(r"(\w+)\(", s).group(1)
                    if stat_cls:
                        comment, key = self.comment_for_member(stat_cls, fn)
                    else:
                        comment, key = self.comment_for_free(fn)
            elif ctx and ctx != "module":
                g = re.match(r"get (\w+)\(\):", s)
                mth = re.match(r"(\w+)\((.*)\):", s)
                prop = re.match(r"(?:readonly )?(\w+)\??:", s)
                nm = g or mth or prop
                if nm and s.endswith(";"):
                    comment, key = self.comment_for_member(ctx, nm.group(1))
                if s == "}":
                    ctx = None

            if comment and not prev_doc:
                out.append(normalize(comment, indent) or "")
                self.hits += 1
            elif key and not prev_doc and re.search(r"[\w>]\s*\(|:\s|\{$", s) and s not in ("};", "}"):
                self.misses.append(key)
            out.append(line)
        return "\n".join(out) + ("\n" if src.endswith("\n") else "")


def main():
    ap = argparse.ArgumentParser(description="Document bindings.d.mts with the MeshLib C++ doc comments")
    ap.add_argument("tsd", help="the emscripten --emit-tsd .d.mts to enrich (in place)")
    ap.add_argument("--bindings", required=True, help="dir with embind source/MRWasmModule/*.cpp")
    ap.add_argument("--comments", default=DEFAULT_XML_DIR,
                    help=f"Doxygen XML dir from scripts/npm/Doxyfile (default: {DEFAULT_XML_DIR})")
    ap.add_argument("--output", default=None, help="output path, or '-' for stdout (default: in place)")
    args = ap.parse_args()

    db, by_tail = load_comment_db(args.comments)
    emb = load_embind_map(args.bindings)
    inj = Injector(db, by_tail, emb)
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
