#!/usr/bin/env bash
# Builds the arXiv submission  arxiv_submission.zip :
#   paper.tex, paper.bbl (arXiv does not run BibTeX), figs/*.png, and the ancillary files anc/ (a copy of the folder Mellin
#   without the paper sources, the original notebooks, the Max Planck data and the cache), then compiles the unpacked archive
#   with pdflatex only, as arXiv does.
set -euo pipefail
cd "$(dirname "$0")"
M=..
OUT=arxiv_submission
rm -rf "$OUT" "$OUT.zip" && mkdir -p "$OUT/figs" "$OUT/anc"

# 1. bibliography
pdflatex -interaction=nonstopmode paper.tex >/dev/null
bibtex paper >/dev/null
pdflatex -interaction=nonstopmode paper.tex >/dev/null
pdflatex -interaction=nonstopmode paper.tex >/dev/null
cp paper.tex paper.bbl "$OUT/"

# 2. the figures actually included
for f in $(grep -o '\\includegraphics\(\[[^]]*\]\)\?{[^}]*}' paper.tex | sed 's/.*{\(.*\)}/\1/'); do
  cp "figs/$f" "$OUT/figs/"
done

# 3. ancillary files
cp "$M/README.md" "$M/ABCInterpolator.wl" "$M/MellinOscillationsThimbles.wl" "$M/MellinOscillationsThimbles.nb" \
   "$M/make_driver_nb.py" "$OUT/anc/"
mkdir -p "$OUT/anc/python" "$OUT/anc/results"
cp "$M"/python/*.py "$M"/python/*.json "$M"/python/*.csv "$M"/python/*.sh "$M"/python/requirements.txt "$OUT/anc/python/"
cp "$M"/results/*.json "$M"/results/*.csv "$M"/results/*.png "$OUT/anc/results/"

# 4. archive, and a test compilation of the unpacked archive without BibTeX
(cd "$OUT" && zip -qr "../$OUT.zip" .)
T=$(mktemp -d); (cd "$T" && unzip -q "$OLDPWD/$OUT.zip" && for i in 1 2 3; do pdflatex -interaction=nonstopmode paper.tex >/dev/null || true; done
  grep -E "^! |undefined|Rerun to get" paper.log && { echo "arXiv test compilation: problems (see above)"; exit 1; } || true
  grep "Output written" paper.log)
rm -rf "$T"
du -h "$OUT.zip"; unzip -l "$OUT.zip" | tail -1
