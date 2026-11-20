# Build the preprint PDF:  make        Package for arXiv upload:  make arxiv
main.pdf: main.tex references.bib figures/*
	pdflatex -interaction=nonstopmode main && bibtex main && pdflatex -interaction=nonstopmode main && pdflatex -interaction=nonstopmode main

arxiv: main.pdf
	tar czf arxiv.tar.gz main.tex main.bbl references.bib figures

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf arxiv.tar.gz

.PHONY: arxiv clean
