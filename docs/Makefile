# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= -W --keep-going
SPHINXBUILD  ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Internal variables
ALLSPHINXOPTS = -d $(BUILDDIR)/doctrees $(SPHINXOPTS) $(SOURCEDIR)

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean livehtml linkcheck coverage doctest

# Clean build directory
clean:
	rm -rf $(BUILDDIR)/*

# Live reload for development
livehtml:
	@echo "Starting live documentation server..."
	@command -v sphinx-autobuild >/dev/null 2>&1 || { \
		echo "sphinx-autobuild not found. Installing..."; \
		pip install sphinx-autobuild; \
	}
	sphinx-autobuild $(SOURCEDIR) $(BUILDDIR)/html \
		--host 0.0.0.0 --port 8000 \
		--ignore "*.swp" --ignore "*.swo" --ignore "*~" \
		--ignore "*.tmp" --ignore ".git/*" \
		$(SPHINXOPTS)

# Check external links
linkcheck:
	@echo "Checking external links..."
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(BUILDDIR)/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output or in $(BUILDDIR)/linkcheck/output.txt."

# Check documentation coverage
coverage:
	@echo "Checking documentation coverage..."
	$(SPHINXBUILD) -b coverage $(ALLSPHINXOPTS) $(BUILDDIR)/coverage
	@echo
	@echo "Testing of coverage in the sources finished, look at the results in $(BUILDDIR)/coverage/python.txt."

# Run doctests
doctest:
	@echo "Running doctests..."
	$(SPHINXBUILD) -b doctest $(ALLSPHINXOPTS) $(BUILDDIR)/doctest
	@echo "doctest finished, look at the results in $(BUILDDIR)/doctest/output.txt."

# Build all formats
all: html pdf epub

# Build PDF (requires LaTeX)
pdf:
	@echo "Building PDF documentation..."
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo "Running LaTeX files through pdflatex..."
	$(MAKE) -C $(BUILDDIR)/latex all-pdf
	@echo "pdflatex finished; the PDF files are in $(BUILDDIR)/latex."

# Development shortcuts
dev: clean html
	@echo "Development build complete."

# Production build with all checks
production: clean
	@echo "Building production documentation with all quality checks..."
	$(SPHINXBUILD) -b html -W -T $(ALLSPHINXOPTS) $(BUILDDIR)/html
	$(MAKE) linkcheck
	$(MAKE) coverage
	@echo "Production build complete."

# Quick development server
serve: html
	@echo "Starting documentation server at http://localhost:8080"
	@cd $(BUILDDIR)/html && python -m http.server 8080

# Check for common issues
check:
	@echo "Checking documentation for common issues..."
	@# Check for TODO/FIXME markers
	@grep -r "TODO\|FIXME" $(SOURCEDIR)/*.rst $(SOURCEDIR)/*/*.rst || echo "No TODO/FIXME markers found."
	@# Check for long lines
	@find $(SOURCEDIR) -name "*.rst" -exec awk 'length > 100 {print FILENAME ":" NR ": " $$0}' {} \;
	@echo "Documentation check complete."

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)