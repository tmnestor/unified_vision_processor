# Testing Strategy - Unified Vision Processor

This testing framework is designed for a dual-environment development workflow:
- **Mac M1**: Fast unit tests and mocked integration tests
- **Multi-GPU Development Machine**: Full integration tests with real models

## Test Categories

### 🚀 Fast Tests (Mac M1 Compatible)
Tests that run quickly on local development machines using smart mocking.

```bash
# Run fast tests only (default on Mac M1)
pytest -m "fast"

# Run all tests except slow/GPU tests (default behavior)
pytest -m "not gpu and not slow"
```

### 🐌 Slow Tests (Development Machine Only)
Tests that require significant computational resources.

```bash
# Run only on development machine
pytest -m "slow"
```

### 🔌 Integration Tests (Development Machine Only)
Tests with real model loading and image processing.

```bash
# Run only on development machine
pytest -m "integration"
```

### 🖥️ GPU Tests (Development Machine Only)
Tests requiring GPU hardware and real model weights.

```bash
# Run only on development machine
pytest -m "gpu"
```

## Running Tests by Component

### CLI Testing
```bash
# Fast CLI tests (Mac M1)
pytest tests/cli/test_cli_fast.py

# All CLI tests except slow ones
pytest tests/cli/ -m "not slow"

# Integration CLI tests (development machine)
pytest tests/cli/test_cli_integration.py -m "integration and gpu"
```

### Computer Vision Testing
```bash
# Fast CV tests with mocking (Mac M1)
pytest tests/computer_vision/ -m "fast"

# Real CV tests (development machine)
pytest tests/computer_vision/test_cv_integration.py -m "integration"
```

### Prompt System Testing
```bash
# Prompt validation tests (Mac M1)
pytest tests/prompts/ -m "fast"
```

## Development Machine Setup

For running integration tests on the development machine:

```bash
# 1. Activate environment
conda activate unified_vision_processor

# 2. Run integration tests
pytest -m "integration and gpu" --tb=long

# 3. Run performance tests
pytest -m "slow" tests/cli/test_cli_integration.py::TestCLIPerformance

# 4. Run all tests (including slow ones)
pytest -m "not fast" 
```

## Local Development (Mac M1)

For fast iteration on Mac M1:

```bash
# 1. Default testing (excludes slow/GPU tests)
pytest

# 2. Fast tests only
pytest -m "fast"

# 3. Specific component testing
pytest tests/cli/test_cli_fast.py -v

# 4. Watch mode for TDD
pytest-watch tests/cli/test_cli_fast.py
```

## Coverage Reporting

```bash
# Generate coverage report
pytest --cov=vision_processor --cov-report=html

# View coverage
open htmlcov/index.html
```

## Test Structure

```
tests/
├── cli/
│   ├── test_cli_fast.py           # Mac M1 compatible CLI tests
│   ├── test_cli_integration.py    # Real model CLI tests
│   └── conftest.py               # CLI test fixtures
├── computer_vision/
│   ├── test_cv_fast.py           # Mocked CV tests
│   └── test_cv_integration.py    # Real CV tests
├── prompts/
│   └── test_prompt_fast.py       # Prompt system tests
└── conftest.py                   # Global test fixtures
```

## CI/CD Pipeline

### GitHub Actions (or similar)

For Mac runners:
```yaml
- name: Run fast tests
  run: pytest -m "fast"
```

For GPU runners:
```yaml
- name: Run integration tests
  run: pytest -m "integration and gpu"
```

## Test Quality Guidelines

### Fast Tests Should:
- ✅ Run in <5 seconds total
- ✅ Use comprehensive mocking
- ✅ Test business logic and interfaces
- ✅ Validate error handling
- ✅ Check parameter validation

### Integration Tests Should:
- ✅ Use real models and components
- ✅ Process actual images
- ✅ Validate end-to-end workflows
- ✅ Test performance benchmarks
- ✅ Verify cross-model consistency

### Both Should:
- ✅ Follow AAA pattern (Arrange, Act, Assert)
- ✅ Have descriptive test names
- ✅ Include proper cleanup
- ✅ Use appropriate fixtures
- ✅ Test error conditions

## Performance Targets

### Fast Tests (Mac M1)
- Total test suite: <30 seconds
- Individual test: <1 second
- Memory usage: <500MB increase

### Integration Tests (Development Machine)
- CLI processing: >20 docs/minute
- CV processing: <2 seconds per image
- Memory usage: <2GB total

## Troubleshooting

### Common Issues

1. **Tests timing out on Mac M1**
   ```bash
   # Make sure you're not running GPU tests
   pytest -m "not gpu and not slow"
   ```

2. **Import errors in tests**
   ```bash
   # Ensure environment is activated
   source /opt/homebrew/Caskroom/miniforge/base/bin/activate unified_vision_processor
   ```

3. **Mock not working as expected**
   ```bash
   # Check mock is patching the right import path
   # Use absolute imports in patch decorators
   ```

### Environment Variables

```bash
# For testing
export TESTING_MODE=true
export VISION_MODEL_PATH_OVERRIDE=/path/to/test/models
export KMP_DUPLICATE_LIB_OK=TRUE
```

## Contributing

When adding new tests:

1. **Start with fast tests** that can run on Mac M1
2. **Add appropriate markers** (`@pytest.mark.fast`, `@pytest.mark.gpu`, etc.)
3. **Use existing fixtures** from conftest.py when possible
4. **Add integration tests** for critical paths (run on development machine)
5. **Update this README** if adding new test categories

## Git Workflow

The testing strategy supports the git-based sync workflow:

1. **Mac M1**: Write and validate fast tests
2. **Git push**: Sync to development machine
3. **Development Machine**: Run integration tests
4. **Git push**: Sync results back

This ensures comprehensive testing while maintaining development velocity.