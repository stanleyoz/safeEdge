# Build the preprint PDF:  make        Package for arXiv upload:  make arxiv
# Uses the texlive Docker image (same toolchain as gazebo-harmonic's papers),
# so no local TeX install is needed.
LATEX = docker run --rm -u $$(id -u):$$(id -g) -v "$$PWD":/w -w /w texlive/texlive:latest

main.pdf: main.tex references.bib $(wildcard figures/*)
	$(LATEX) latexmk -pdf -interaction=nonstopmode main.tex

arxiv: main.pdf
	tar czf arxiv.tar.gz main.tex main.bbl references.bib figures

clean:
	$(LATEX) latexmk -C main.tex
	rm -f main.bbl arxiv.tar.gz build.log

.PHONY: arxiv clean
