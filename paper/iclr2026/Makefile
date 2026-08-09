.PHONY: all figures validate clean

all: figures
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	python3 validate_claims.py

figures:
	python3 figures/gen_figures.py

validate:
	python3 validate_claims.py

clean:
	latexmk -C main.tex
