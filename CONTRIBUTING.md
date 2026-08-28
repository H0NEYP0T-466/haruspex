# Contributing to HARUSPEX

Thank you for your interest in contributing to HARUSPEX! This document provides guidelines and instructions for contributing.

## How to Contribute

### 1. Fork the Repository

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/<your-username>/haruspex.git
cd haruspex
git remote add upstream https://github.com/H0NEYP0T-466/haruspex.git
```

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/add-new-explainer`
- `fix/correlation-scoring-bug`
- `docs/update-api-endpoints`
- `refactor/model-loading`

### 3. Set Up Development Environment

**Backend:**
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .  # Install in editable mode
```

**Frontend:**
```bash
# From project root
npm install
```

### 4. Make Your Changes

- Follow the code style guidelines below
- Write tests for new functionality
- Update documentation as needed

### 5. Commit Your Changes

```bash
git add .
git commit -m "feat: add new correlation scoring algorithm"
```

Use [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` — New feature
- `fix:` — Bug fix
- `docs:` — Documentation changes
- `refactor:` — Code refactoring
- `perf:` — Performance improvement
- `test:` — Adding/updating tests
- `build:` — Build system changes
- `ci:` — CI configuration changes
- `chore:` — Maintenance tasks

### 6. Push and Create a Pull Request

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub against the `main` branch.

---

## Code Style

### Python (Backend)

- **Formatter/Linter:** Ruff (configured in `backend/pyproject.toml`)
- **Line length:** 100 characters
- **Style:** PEP 8 with type hints
- **Quotes:** Double quotes
- **Indentation:** 4 spaces

```bash
cd backend
ruff check .        # Lint
ruff format .       # Format
```

Key rules enabled: `E`, `F`, `W`, `I`, `N`, `UP`, `B`, `A`, `SIM`

### TypeScript (Frontend)

- **Linter:** Oxlint
- **Formatter:** Vite built-in

```bash
npm run lint        # Lint
npm run build       # Build check
```

### General Guidelines

- Write docstrings for all public functions and classes
- Use meaningful variable and function names
- Keep functions focused and small (< 50 lines where possible)
- Avoid premature optimization
- Follow existing patterns in the codebase

---

## Testing

### Running Tests

```bash
cd backend
pytest tests/ -v
```

### Writing Tests

- Place unit tests in `backend/tests/unit/`
- Place integration tests in `backend/tests/integration/`
- Name test files `test_<module>.py`
- Name test functions `test_<description>()`
- Use descriptive test names

```python
def test_correlator_maps_feature_to_forensic_evidence():
    """Test that the correlator correctly maps SHAP attributions to YARA findings."""
    # Arrange
    correlator = EvidenceCorrelator()
    # Act
    result = correlator.correlate(shap_values, forensic_evidence)
    # Assert
    assert len(result) > 0
    assert all(e.severity > 0 for e in result)
```

---

## Bug Reports

When filing a bug report, please include:

1. **Summary** — Clear description of the problem
2. **Steps to Reproduce** — Minimal steps to trigger the bug
3. **Expected Behavior** — What should happen
4. **Actual Behavior** — What actually happens
5. **Environment** — OS, Python version, Node version, package versions
6. **Logs/Screenshots** — Any relevant error output

Use the [Bug Report template](https://github.com/H0NEYP0T-466/haruspex/issues/new?template=bug_report.yml).

---

## Feature Requests

When proposing a feature, please include:

1. **Problem Statement** — What problem does this solve?
2. **Proposed Solution** — How should it work?
3. **Alternatives Considered** — Other approaches you evaluated
4. **Scope** — What's in and out of scope
5. **Risks** — Potential downsides or trade-offs

Use the [Feature Request template](https://github.com/H0NEYP0T-466/haruspex/issues/new?template=feature_request.yml).

---

## Documentation

- Update `README.md` if adding/changing features or API endpoints
- Add docstrings to new functions and classes
- Update inline comments for complex logic
- Include examples for new API endpoints

---

## Areas for Contribution

Here are some areas where contributions are especially welcome:

- **New XAI methods** — Integrate additional explanation techniques
- **New YARA rules** — Expand forensic detection capabilities
- **Model architectures** — Implement new deep learning models
- **Evaluation metrics** — Add domain-specific malware detection metrics
- **Frontend improvements** — Dashboard visualizations, UX enhancements
- **Documentation** — Tutorials, examples, API docs
- **Performance** — Optimization of prediction/explanation pipelines
- **Testing** — Increase test coverage

---

## Questions?

If you have questions about contributing, feel free to open a [Discussion](https://github.com/H0NEYP0T-466/haruspex/discussions) or reach out to the maintainers.

Thank you for contributing to HARUSPEX!
