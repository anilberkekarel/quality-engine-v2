arXiv submission package
========================

Title : Multi-Sensor Latent-State Fusion for Fundamental Analysis:
        A Pre-Registered, Point-in-Time Evaluation in Semiconductors
Author: Anil Berke Karel (Politecnico di Torino)

Contents
--------
main.tex          - the paper (single-column, 11pt, article class)
references.bib     - BibTeX source for the 7 references
main.bbl           - pre-compiled bibliography (REQUIRED: arXiv does not run BibTeX)
figures/*.pdf      - the 10 figures (vector PDF, referenced by \includegraphics)

Document class & packages (all standard / arXiv-safe):
  article, inputenc, fontenc, lmodern, geometry, amsmath, amssymb,
  graphicx, booktabs, caption, natbib (round,authoryear), hyperref.

How arXiv builds it
-------------------
arXiv runs pdflatex (using the bundled main.bbl). No BibTeX pass is needed
because main.bbl is included. If you rebuild locally from scratch:

  pdflatex main
  bibtex   main
  pdflatex main
  pdflatex main

Output: main.pdf, 20 pages, US Letter.
