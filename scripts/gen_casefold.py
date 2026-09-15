# Generates MRUnicodeCaseFold.inl: a sorted simple-case-folding table from CaseFolding.txt.
# Simple folding = status C (common) + S (simple), each a 1:1 codepoint mapping.
# Status F (full, 1:many) and T (Turkic) are intentionally excluded.
import sys

src, out = sys.argv[1], sys.argv[2]
version = None
pairs = []
for line in open(src, encoding='utf-8'):
    line = line.rstrip('\n')
    if line.startswith('# CaseFolding-'):
        version = line[2:].strip()
    if not line or line.startswith('#'):
        continue
    parts = [p.strip() for p in line.split('#')[0].split(';')]
    if len(parts) < 3:
        continue
    code, status, mapping = parts[0], parts[1], parts[2]
    if status not in ('C', 'S'):
        continue
    frm = int(code, 16)
    tos = mapping.split()
    assert len(tos) == 1, f"simple fold must be 1:1, got {line!r}"
    pairs.append((frm, int(tos[0], 16)))

pairs.sort(key=lambda p: p[0])
# sanity: strictly increasing, no ASCII surprises
for i in range(1, len(pairs)):
    assert pairs[i][0] > pairs[i-1][0], f"dup/unsorted at {pairs[i]}"

with open(out, 'w', encoding='utf-8', newline='\n') as f:
    f.write(f"// Generated from {version} (https://www.unicode.org/Public/UNIDATA/CaseFolding.txt)\n")
    f.write("// Simple case folding: status C + S entries (1:1 codepoint mappings). Do not edit by hand.\n")
    f.write("// Regenerate with scripts/gen_casefold.py.\n")
    f.write(f"// {len(pairs)} entries.\n")
    f.write("{\n")
    for frm, to in pairs:
        f.write(f"    {{ 0x{frm:04X}, 0x{to:04X} }},\n")
    f.write("}\n")

# sanity-check a few known foldings so a format/parse regression fails loudly
expected = { 0x0041: 0x0061, 0x0412: 0x0432, 0x0391: 0x03B1, 0x212A: 0x006B, 0x10412: 0x1043A }
folds = dict( pairs )
for frm, to in expected.items():
    assert folds.get( frm ) == to, f"unexpected fold U+{frm:04X} -> {folds.get(frm)} (expected U+{to:04X})"

print( f"wrote {out}: {len(pairs)} entries, version {version}" )
