#!/usr/bin/env bash
# Diagnostic: which template instantiations are duplicated across the binding fragment object files?
# Every COMDAT (weak) symbol defined in more than one fragment is work the compiler did more than once;
# `count * size` ranks the waste. Usage: analyze_fragment_dups.sh <temp output dir> [min fragment count]
set -euo pipefail
dir="${1:?usage: analyze_fragment_dups.sh <temp output dir> [min fragment count]}"
min="${2:-2}"
shopt -s nullglob
files=( "$dir"/mrmesh.fragment.*.o )
echo "Analyzing ${#files[@]} fragment object files in $dir"
if [[ ${#files[@]} -eq 0 ]]; then echo "No fragment objects found, nothing to do."; exit 0; fi
NM=$(command -v llvm-nm || command -v nm)
raw=$(mktemp)

for f in "${files[@]}"; do
    "$NM" --defined-only --print-size --radix=d "$f"
done | awk '$3 ~ /^[WVu]$/ && NF == 4 { print $2, $4 }' | awk -v MIN="$min" '
    { cnt[$2]++; sz[$2] = $1 }
    END { for (s in cnt) if (cnt[s] >= MIN) printf "%6d %14d %s\n", cnt[s], cnt[s]*sz[s], s }
' | sort -k2,2 -rn > "$raw"

echo "Symbols duplicated across >= $min fragments: $(wc -l < "$raw")"
echo "Total duplicated COMDAT bytes (sum of count*size): $(awk '{ t += $2 } END { print t }' "$raw")"
echo
echo "=== Histogram: how many symbols appear in how many fragments (count, symbols):"
awk '{ print $1 }' "$raw" | sort -n | uniq -c | tail -20
echo
echo "=== Aggregate by origin (first match wins):"
c++filt < "$raw" | awk '
    {
        name = $0; sub(/^ *[0-9]+ +[0-9]+ +/, "", name)
        o = "other"
        if      (index(name, "pybind11::")) o = "pybind11"
        else if (index(name, "MRBind::"))   o = "mrbind"
        else if (index(name, "MR::"))       o = "MR"
        else if (name ~ /^(std|__gnu|operator)/) o = "std"
        b[o] += $2; n[o]++
    }
    END { for (o in b) printf "%-10s %8d symbols %16d bytes\n", o, n[o], b[o] }
' | sort -k4,4 -rn
echo
echo "=== Top 100 by total duplicated bytes (fragments, total bytes, symbol):"
head -100 "$raw" | c++filt | cut -c1-300
echo
echo "=== Top 60 among symbols present in (almost) every fragment (>= 24):"
awk '$1 >= 24' "$raw" | head -60 | c++filt | cut -c1-300
rm -f "$raw"
