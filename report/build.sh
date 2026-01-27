#!/bin/bash
set -e

pdflatex -interaction=nonstopmode report.tex

# Run bibtex only if citations exist (i.e., if report.aux contains \citation)
if grep -q "\\citation" report.aux; then
  bibtex report
fi

pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex

echo "Build successful: report.pdf"

