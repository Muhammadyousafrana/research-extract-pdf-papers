# ─────────────────────────────────────────────────────────────────────
# Research Server — Dev Automation
# Usage: make <target>
# ─────────────────────────────────────────────────────────────────────

PYTHON           = python
SRC              = research_server.py
TESTS            = test_server.py
PYLINT_THRESHOLD = 7.0

.PHONY: all format lint test coverage clean help

# Default — format, lint, then test
all: format lint test

## format: Auto-format code with black
format:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  BLACK — $(SRC) $(TESTS)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	$(PYTHON) -m black $(SRC) $(TESTS)

## format-check: Check formatting without making changes (CI-safe)
format-check:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  BLACK CHECK — $(SRC) $(TESTS)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	$(PYTHON) -m black --check $(SRC) $(TESTS)

## lint: Run pylint on research_server.py (fails if score < 7.0)
lint:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  PYLINT — $(SRC)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	$(PYTHON) -m pylint $(SRC) \
		--disable=C0114,C0115,C0116 \
		--fail-under=$(PYLINT_THRESHOLD)

## test: Run pytest with verbose output
test:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  PYTEST — $(TESTS)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	$(PYTHON) -m pytest $(TESTS) -v --tb=short

## coverage: Run tests with coverage report
coverage:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  COVERAGE REPORT"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	$(PYTHON) -m pytest $(TESTS) -v \
		--cov=research_server \
		--cov-report=term-missing \
		--cov-report=html:htmlcov

## clean: Remove cache, coverage, and temp files (Windows compatible)
clean:
	@echo "Cleaning..."
	$(PYTHON) -c "\
import shutil, os; \
[shutil.rmtree(p, ignore_errors=True) for p in ['__pycache__', '.pytest_cache', 'htmlcov']]; \
os.remove('.coverage') if os.path.exists('.coverage') else None; \
[os.remove(os.path.join(r,f)) for r,_,fs in os.walk('.') for f in fs if f.endswith('.pyc')]; \
print('Done.')"

## help: Show available targets
help:
	@echo "Available targets:"
	@echo "  make              — format + lint + test (default)"
	@echo "  make format       — auto-format with black"
	@echo "  make format-check — check formatting without changes (CI)"
	@echo "  make lint         — pylint with score threshold $(PYLINT_THRESHOLD)"
	@echo "  make test         — pytest verbose"
	@echo "  make coverage     — pytest + coverage report + htmlcov/"
	@echo "  make clean        — remove cache and temp files"