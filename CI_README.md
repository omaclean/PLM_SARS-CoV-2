# Continuous Integration Setup

This document describes the CI/CD setup for automated testing of the PLM_SARS-CoV-2 repository.

## Overview

The repository uses **GitHub Actions** to automatically run tests on every push and pull request. This ensures code quality and catches bugs early.

## What Gets Tested

### On Every Push/PR to `main` or `develop`:

1. **Unit Tests** - All functions in `Functions_HuggingFace.py` are tested
2. **Integration Tests** - Multi-function workflows are validated
3. **Code Coverage** - Test coverage metrics are generated
4. **Code Linting** - Style and syntax checks with flake8

### Test Matrix

Tests run on multiple Python versions to ensure compatibility:
- Python 3.8
- Python 3.9
- Python 3.10
- Python 3.11

## Workflow Files

### `.github/workflows/tests.yml`

Main CI pipeline with two jobs:

#### 1. Test Job
- Runs on Ubuntu latest
- Tests across Python 3.8-3.11
- Installs dependencies
- Runs pytest with coverage
- Uploads coverage to Codecov

#### 2. Lint Job
- Runs flake8 for syntax errors
- Checks formatting with black
- Continues even if linting issues are found (warnings only)

## Running Tests Locally

### Quick Start

```bash
# Run all tests
./run_tests.sh

# Run with verbose output
./run_tests.sh -v

# Run with coverage report
./run_tests.sh -c

# Run tests in parallel (faster)
./run_tests.sh -p

# Run specific test class
./run_tests.sh -t TestMutationFunctions

# All options combined
./run_tests.sh -v -c -p
```

### Manual pytest Commands

```bash
# Basic test run
pytest tests/test_functions_huggingface.py

# Verbose with short traceback
pytest tests/test_functions_huggingface.py -v --tb=short

# With coverage
pytest tests/test_functions_huggingface.py --cov=Functions_HuggingFace --cov-report=html

# Run specific test
pytest tests/test_functions_huggingface.py::TestMutationFunctions::test_mutate_sequence_single_mutation

# Stop after first failure
pytest tests/test_functions_huggingface.py -x

# Run tests matching pattern
pytest tests/test_functions_huggingface.py -k "mutation"
```

## Setting Up CI for Your Fork

If you fork this repository, GitHub Actions will automatically be enabled. No additional setup is required.

### Optional: Enable Codecov

1. Go to [codecov.io](https://codecov.io)
2. Sign in with GitHub
3. Add your repository
4. Coverage reports will be automatically uploaded

## Badge Status

Add these badges to your README.md to show CI status:

```markdown
![Tests](https://github.com/omaclean/PLM_SARS-CoV-2/workflows/Run%20Tests/badge.svg)
[![codecov](https://codecov.io/gh/omaclean/PLM_SARS-CoV-2/branch/main/graph/badge.svg)](https://codecov.io/gh/omaclean/PLM_SARS-CoV-2)
```

## Test Configuration

### pytest.ini

Configuration file defining:
- Test discovery patterns
- Output formatting
- Coverage settings
- Test markers for categorization

### Test Markers

Organize tests with markers:

```python
@pytest.mark.unit
def test_something():
    pass

@pytest.mark.slow
def test_slow_operation():
    pass

@pytest.mark.requires_model
def test_with_esm_model():
    pass
```

Run specific markers:
```bash
pytest -m unit          # Run only unit tests
pytest -m "not slow"    # Skip slow tests
```

## Dependencies

Tests require:
```
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-xdist>=3.0.0  # For parallel testing
biopython>=1.79
pandas>=1.3.0
numpy>=1.21.0
torch>=1.9.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## Troubleshooting

### Tests Fail Locally But Pass in CI

- Ensure you have all dependencies installed
- Check Python version matches CI (3.8-3.11)
- Clear pytest cache: `pytest --cache-clear`

### Tests Pass Locally But Fail in CI

- Check for platform-specific issues
- Verify all required files are committed
- Check for absolute path dependencies

### Slow Tests

- Run in parallel: `./run_tests.sh -p`
- Skip slow tests: `pytest -m "not slow"`
- Profile slow tests: `pytest --durations=10`

## Best Practices

### Before Committing

1. Run tests locally: `./run_tests.sh -v -c`
2. Check coverage: Aim for 80%+ on new code
3. Fix any linting issues
4. Ensure all tests pass

### Writing New Tests

1. Add tests for new functions immediately
2. Test both success and failure cases
3. Include edge cases and boundary conditions
4. Use descriptive test names
5. Add docstrings explaining what is tested

### Maintaining Tests

- Keep tests independent (no shared state)
- Use fixtures for common setup
- Mock external dependencies (models, files)
- Update tests when changing function signatures

## Coverage Goals

| Component | Target Coverage |
|-----------|----------------|
| Mutation Functions | 90%+ |
| Translation Functions | 90%+ |
| Utility Functions | 85%+ |
| I/O Functions | 80%+ |
| Overall | 80%+ |

View current coverage:
```bash
./run_tests.sh -c
open htmlcov/index.html  # View detailed coverage report
```

## Continuous Improvement

The test suite is continuously improved:
- Add tests for newly discovered bugs
- Increase coverage for untested code paths
- Add integration tests for common workflows
- Performance tests for critical paths

## Contact

For questions about testing or CI:
- Open an issue on GitHub
- Check existing test examples
- Refer to pytest documentation

## References

- [pytest documentation](https://docs.pytest.org/)
- [GitHub Actions documentation](https://docs.github.com/en/actions)
- [Codecov documentation](https://docs.codecov.com/)
