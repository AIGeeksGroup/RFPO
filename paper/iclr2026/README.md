# Paper build

From the repository root:

```bash
python3 paper/iclr2026/figures/gen_figures.py
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/iclr2026/main.tex
python3 paper/iclr2026/validate_claims.py
```

The figure script reads committed result JSON from `experiments/` and writes
vector PDF plus 300-DPI PNG files. `main.pdf` is the anonymous review draft.

The ICLR 2026 style is used as a complete, unmodified formatting shell. The
scientific framing is workshop/short-paper strength until further cross-task or
real-robot evidence exists; selecting a venue may require changing the template
and page budget without changing claims.
