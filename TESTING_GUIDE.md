# Testing Guide - Unified Vision Processor

This guide provides comprehensive instructions for running tests in the unified vision processor project.

## Test Structure Overview

The project uses pytest as the testing framework with the following structure:

- **Unit Tests**: `tests/unit/` - 6 test files covering core components
- **Integration Tests**: `tests/integration/` - Production readiness tests
- **Banking Tests**: `tests/banking/` - 5 test files for banking functionality  
- **Performance Tests**: `tests/performance/` - Performance validation
- **Evaluation Tests**: `tests/evaluation/` - Framework evaluation

## Test Categories

### Unit Tests (`tests/unit/`)
- `test_ato_compliance.py` - ATO compliance validation
- `test_evaluation_framework.py` - Evaluation system testing
- `test_hybrid_extraction_manager.py` - Extraction pipeline tests
- `test_model_factory.py` - Model factory and configuration
- `test_model_fairness.py` - Model fairness validation
- `test_unified_config.py` - Configuration management

### Banking Tests (`tests/banking/`)
- `test_bank_recognizer.py` - Australian bank recognition
- `test_bank_statement_cv.py` - Computer vision for bank statements
- `test_bank_statement_handler.py` - Bank statement processing
- `test_bsb_validator.py` - BSB (Bank State Branch) validation
- `test_transaction_categorizer.py` - Transaction categorization

### Integration Tests (`tests/integration/`)
- `test_production_readiness.py` - End-to-end production validation

### Performance Tests (`tests/performance/`)
- `test_performance_validation.py` - Performance benchmarking

## Running Tests on Remote Environment (2x H200)

### Prerequisites
Ensure you're on the remote development environment with:
```bash
conda activate unified_vision_processor
```

### Basic Commands

#### 1. Test Discovery
```bash
# See all available tests without running them
pytest --collect-only
```

#### 2. Run All Tests
```bash
# Run complete test suite with coverage
pytest tests/ -v --cov=vision_processor --cov-report=html

# Run all tests with basic output
pytest tests/ -v
```

#### 3. Run Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/ -v

# Banking functionality tests
pytest tests/banking/ -v

# Integration tests
pytest tests/integration/ -v

# Performance tests
pytest tests/performance/ -v
```

#### 4. Run Individual Test Files
```bash
# Configuration tests
pytest tests/unit/test_unified_config.py -v

# Bank recognition tests
pytest tests/banking/test_bank_recognizer.py -v

# ATO compliance tests
pytest tests/unit/test_ato_compliance.py -v

# Production readiness tests
pytest tests/integration/test_production_readiness.py -v
```

#### 5. Detailed Output Options
```bash
# Verbose output with stdout capture
pytest tests/ -v -s --tb=short

# Show local variables in tracebacks
pytest tests/ -v --tb=long

# Stop on first failure
pytest tests/ -x

# Run specific test by name pattern
pytest tests/ -k "test_bank" -v
```

#### 6. Coverage Reports
```bash
# Generate HTML coverage report
pytest tests/ --cov=vision_processor --cov-report=html

# Generate terminal coverage report
pytest tests/ --cov=vision_processor --cov-report=term-missing

# Coverage for specific modules
pytest tests/banking/ --cov=vision_processor.banking --cov-report=term
```

## Expected Issues and Solutions

### Common Issues to Address

1. **Import Errors**
   - Missing dependencies in test environment
   - Incorrect PYTHONPATH configuration
   - Module import failures

2. **Model Loading Failures**
   - Use mock models for testing (configured in `conftest.py`)
   - Avoid loading actual vision models during unit tests
   - Configure CPU-only testing for faster execution

3. **GPU Memory Issues**
   - Configure tests to use CPU by default
   - Use smaller batch sizes for testing
   - Mock GPU-intensive operations

4. **Missing Test Data**
   - Ensure test datasets are available
   - Use mock data from `conftest.py` fixtures
   - Check for required image files

### Test Configuration

The `tests/conftest.py` file provides:
- Mock models and configurations
- Sample test data and fixtures
- CPU-only configuration for testing
- Australian business test data
- Valid/invalid ABN samples

### Debugging Failed Tests

```bash
# Run with Python debugger
pytest tests/unit/test_unified_config.py --pdb

# Run with detailed error information
pytest tests/ -v --tb=long --capture=no

# Run only failed tests from last run
pytest --lf

# Show slowest tests
pytest tests/ --durations=10
```

## Recommended Testing Workflow

### Phase 1: Basic Validation
```bash
# 1. Check test discovery
pytest --collect-only

# 2. Run unit tests first
pytest tests/unit/ -v

# 3. Check configuration tests specifically
pytest tests/unit/test_unified_config.py -v
```

### Phase 2: Component Testing
```bash
# 4. Test banking components
pytest tests/banking/ -v

# 5. Test model factory
pytest tests/unit/test_model_factory.py -v

# 6. Test extraction pipeline
pytest tests/unit/test_hybrid_extraction_manager.py -v
```

### Phase 3: Integration Testing
```bash
# 7. Run integration tests
pytest tests/integration/ -v

# 8. Run performance validation
pytest tests/performance/ -v
```

### Phase 4: Full Validation
```bash
# 9. Complete test suite with coverage
pytest tests/ -v --cov=vision_processor --cov-report=html

# 10. Review coverage report
# Open htmlcov/index.html in browser
```

## Test Data and Fixtures

The testing framework includes:
- **Australian Test Businesses**: Woolworths, Coles, JB Hi-Fi, etc.
- **Valid ABNs**: Real Australian Business Numbers for testing
- **Sample Documents**: Mock receipts, invoices, bank statements
- **Test Images**: Synthetic image data for computer vision tests
- **Ground Truth Data**: Expected extraction results

## Continuous Integration

For automated testing:
```bash
# Run tests suitable for CI
pytest tests/ --cov=vision_processor --cov-report=xml --junitxml=test-results.xml

# Quick smoke tests
pytest tests/unit/test_unified_config.py tests/unit/test_model_factory.py -v
```

## Getting Help

If tests fail:
1. Check the error messages for missing dependencies
2. Verify the conda environment is activated
3. Ensure vision_processor package is importable
4. Review the test configuration in `conftest.py`
5. Run tests in isolation to identify specific issues

## Status Update

✅ **Import Issues Fixed**: All 4 import errors have been resolved:
- `QualityGrade` and `ProcessingStage` added to `pipeline_components.py`
- `AmountValidator` added to `field_validators.py` 
- `ModelCreationError` added to `model_factory.py`

**Next Steps for Remote Environment:**

1. **Sync the fixes**: Pull the latest changes to your remote environment
2. **Re-run test discovery**: `pytest --collect-only` (should now find all 136 tests)
3. **Start with unit tests**: `pytest tests/unit/ -v`

Start with `pytest --collect-only` to verify test discovery, then proceed with unit tests before moving to integration testing.