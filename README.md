# Unified Vision Document Processing Architecture

A comprehensive system for consolidating InternVL PoC and Llama-3.2 vision implementations into a single, model-agnostic document processing package focused on Australian tax documents.

## 🎯 Project Status: Complete Implementation

This project represents a complete, production-ready unified vision processing system with all 8 phases successfully implemented.

### Implementation Overview

✅ **Phase 1: Foundation Setup** - Complete  
✅ **Phase 2: Model Integration** - Complete  
✅ **Phase 3: Llama Pipeline Integration** - Complete  
✅ **Phase 4: Feature Integration** - Complete  
✅ **Phase 5: Handler and Prompt Integration** - Complete  
✅ **Phase 6: Evaluation Framework** - Complete  
✅ **Phase 7: CLI and Production Features** - Complete  
✅ **Phase 8: Testing and Validation** - Complete  

## 🏗️ Architecture Foundation

The unified system uses Llama-3.2's 7-step processing pipeline as its foundation, integrating InternVL's advanced technical capabilities:

```
Vision Document → Classification → Model Inference → Primary Extraction 
                                                          ↓
                  Confidence ← ATO Compliance ← Validation ← AWK Fallback
```

### Core Components

- **BaseVisionModel**: Unified interface for InternVL3 and Llama-3.2-Vision
- **ModelFactory**: Multi-GPU optimization and device management
- **UnifiedConfig**: Environment-driven configuration system
- **HybridExtractionManager**: 7-step processing pipeline
- **ATOComplianceValidator**: Australian tax compliance validation
- **UnifiedEvaluator**: Comprehensive evaluation framework

## 🚀 Key Features

### Model Agnostic Processing
- Business logic completely independent of vision model choice
- Seamless switching between InternVL3 and Llama-3.2-Vision
- Standardized response format across models

### Australian Tax Focus
- 11 Australian tax document types supported
- Complete ATO compliance validation
- GST calculation verification (10% standard rate)
- ABN validation and format checking
- Australian banking integration (BSB validation)

### Production Readiness
- 5-level production readiness assessment
- Multi-GPU optimization for development (2x H200)
- Single GPU optimization for production (V100 16GB)
- Graceful degradation with multi-tier fallbacks
- Comprehensive error handling and logging

### Advanced Processing
- Computer vision highlight detection
- AWK fallback extraction (2,000+ rules)
- 4-component confidence scoring
- Cross-model evaluation framework
- Batch processing with parallel workers

## 📦 Package Structure

```
unified_vision_processor/
├── vision_processor/                 # Main package
│   ├── config/                      # Unified configuration
│   │   ├── unified_config.py        # Environment-driven config
│   │   └── model_factory.py         # Model instantiation
│   ├── models/                      # Model abstractions
│   │   ├── base_model.py            # BaseVisionModel interface
│   │   ├── internvl_model.py        # InternVL3 implementation
│   │   └── llama_vision_model.py    # Llama-3.2-Vision implementation
│   ├── classification/              # Document classification
│   │   └── australian_tax_classifier.py
│   ├── extraction/                  # 7-step processing pipeline
│   │   ├── hybrid_extraction_manager.py
│   │   ├── pipeline_components.py
│   │   └── awk_extraction.py
│   ├── confidence/                  # Confidence scoring
│   │   └── confidence_integration_manager.py
│   ├── compliance/                  # ATO compliance validation
│   │   └── ato_compliance_validator.py
│   ├── handlers/                    # Document-specific processors
│   │   ├── fuel_receipt_handler.py
│   │   ├── tax_invoice_handler.py
│   │   └── [9 other handlers]
│   ├── computer_vision/             # InternVL highlight detection
│   │   └── highlight_detector.py
│   ├── evaluation/                  # Cross-model evaluation
│   │   ├── unified_evaluator.py
│   │   ├── sroie_evaluator.py
│   │   ├── model_comparator.py
│   │   └── report_generator.py
│   ├── prompts/                     # 60+ specialized prompts
│   │   └── prompt_library.py
│   ├── cli/                         # Command line interfaces
│   │   ├── unified_cli.py
│   │   └── batch_processing.py
│   ├── banking/                     # Australian banking integration
│   │   └── bsb_validator.py
│   └── utils/                       # Shared utilities
│       └── document_utils.py
├── tests/                           # Comprehensive test suite
│   ├── unit/                       # Unit tests
│   ├── integration/                # Integration tests
│   └── performance/                # Performance tests
└── test_phase*.py                  # Phase validation scripts
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU (recommended)
- Conda package manager

### Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd unified_vision_processor

# Create conda environment
conda env create -f environment.yml
conda activate unified_vision_processor

# Install package in development mode (OPTIONAL - only needed for console scripts)
pip install -e .

# Verify installation (OPTIONAL - skip if you can't install)
python -c "import vision_processor; print('✅ Installation successful')"
```

### 🔒 Restricted Environment Setup

If you don't have permission to run `pip install -e .` on your work computer:

**Option 1: Using .env file (recommended)**
```bash
# 1. Create conda environment
conda env create -f environment.yml
conda activate unified_vision_processor

# 2. Create .env file with all configuration (including PYTHONPATH export)
cat >> .env << 'EOF'
# Python Path for package access (export needed for shell)
export PYTHONPATH=${PYTHONPATH}:$(pwd)

# Vision model configuration
VISION_MODEL_TYPE=internvl3
VISION_PROCESSING_PIPELINE=7step
# ... other config options
EOF

# 3. Source the .env file to set PYTHONPATH
source .env

# 4. Verify the package is accessible
python -c "import vision_processor; print('✅ Package accessible via .env PYTHONPATH')"

# 5. Run CLI commands (no installation needed)
python -m vision_processor.cli.unified_cli process datasets/image25.png --model internvl3
```

**Option 2: Temporary PYTHONPATH (quick test)**
```bash
# Set PYTHONPATH temporarily (reset each session)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run CLI directly
python -m vision_processor.cli.unified_cli process datasets/image25.png --model internvl3
```

**What you CAN'T do without installation:**
- Use console scripts like `unified-vision` command
- Import the package from other directories

**What you CAN do without installation:**
- Run all CLI commands using `python -m vision_processor.cli.unified_cli`
- Use the Python API from within the project directory
- All functionality works exactly the same

### Model Configuration

Create a `.env` file for model paths and configuration:

```bash
# Model Paths - CRITICAL FOR OFFLINE PRODUCTION
VISION_INTERNVL_MODEL_PATH=/path/to/InternVL3-8B
VISION_LLAMA_MODEL_PATH=/path/to/Llama-3.2-11B-Vision

# Model Selection
VISION_MODEL_TYPE=internvl3              # internvl3 | llama32_vision
VISION_PROCESSING_PIPELINE=7step         # Llama 7-step pipeline standard
VISION_EXTRACTION_METHOD=hybrid          # hybrid | key_value | awk_only

# Feature Integration
VISION_HIGHLIGHT_DETECTION=true          # InternVL highlight detection
VISION_AWK_FALLBACK=true                # Comprehensive AWK fallback
VISION_GRACEFUL_DEGRADATION=true        # Llama graceful degradation
VISION_CONFIDENCE_COMPONENTS=4           # 4-component confidence scoring

# GPU Configuration
VISION_GPU_MEMORY_LIMIT=15360            # V100 16GB limit with buffer (MB)
VISION_ENABLE_8BIT_QUANTIZATION=true     # Memory optimization for production
VISION_MULTI_GPU_DEV=true               # Enable for 2x H200 development
VISION_SINGLE_GPU_PROD=true             # Target single V100 production

# Offline Mode - Default is true for production safety
VISION_OFFLINE_MODE=true                 # Default: true
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

## 🖥️ Usage

### Command Line Interface

The CLI provides two ways to access commands:

1. **Python Module** (direct): `python -m vision_processor.cli.unified_cli <command>`
2. **Console Script** (after pip install): `unified-vision <command>`

#### Single Document Processing
```bash
# Process single document with InternVL3
python -m vision_processor.cli.unified_cli process datasets/image25.png --model internvl3

# Process with Llama-3.2-Vision and save results
python -m vision_processor.cli.unified_cli process datasets/invoice.png \
    --model llama32_vision \
    --output results.json \
    --verbose

# Process with document type hint
python -m vision_processor.cli.unified_cli process datasets/image25.png \
    --model internvl3 \
    --type fuel_receipt \
    --output result.json

# Using console script (after pip install -e .)
unified-vision process datasets/image25.png --model internvl3
```

#### Batch Processing
```bash
# Process entire directory
python -m vision_processor.cli.unified_cli batch datasets/ \
    --model internvl3 \
    --output results/ \
    --max-documents 100

# Generate comprehensive report
python -m vision_processor.cli.unified_cli batch datasets/ \
    --model llama32_vision \
    --output results/ \
    --generate-report

# Using console script
unified-vision batch datasets/ --model internvl3 --output results/
```

#### Model Comparison
```bash
# Compare InternVL3 vs Llama-3.2-Vision
python -m vision_processor.cli.unified_cli compare datasets/ ground_truth/ \
    --models internvl3,llama32_vision \
    --output comparison_results/ \
    --confidence-threshold 0.7

# Using console script
unified-vision compare datasets/ ground_truth/ --models internvl3,llama32_vision
```

#### SROIE Evaluation
```bash
# Evaluate on SROIE dataset
python -m vision_processor.cli.unified_cli evaluate sroie_dataset/ sroie_ground_truth/ \
    --model internvl3 \
    --output evaluation_results/

# Using console script
unified-vision evaluate sroie_dataset/ sroie_ground_truth/ --model internvl3
```

#### Get Help
```bash
# Show all available commands
python -m vision_processor.cli.unified_cli --help

# Get help for specific command
python -m vision_processor.cli.unified_cli process --help
python -m vision_processor.cli.unified_cli batch --help
python -m vision_processor.cli.unified_cli compare --help
python -m vision_processor.cli.unified_cli evaluate --help
```

### CLI Troubleshooting

#### Common Issues

**❌ ModuleNotFoundError: No module named 'llama_vision'**
```bash
# Wrong (old package structure):
python -m llama_vision.cli.llama_single extract datasets/image25.png

# ✅ Correct (current package structure):
python -m vision_processor.cli.unified_cli process datasets/image25.png
```

**❌ Console scripts not available**
```bash
# Option 1: Install package (if you have permissions):
pip install -e .
unified-vision process datasets/image25.png --model internvl3

# Option 2: Use Python module directly (no installation needed):
python -m vision_processor.cli.unified_cli process datasets/image25.png --model internvl3
```

**❌ Import errors**
```bash
# Make sure you're in the right environment:
conda activate unified_vision_processor

# Verify installation:
python -c "import vision_processor; print('✅ Package available')"
```

### Python API

#### Basic Usage
```python
from vision_processor.config.unified_config import UnifiedConfig, ModelType
from vision_processor.extraction.hybrid_extraction_manager import UnifiedExtractionManager

# Initialize configuration
config = UnifiedConfig()
config.model_type = ModelType.INTERNVL3
config.confidence_threshold = 0.8

# Create extraction manager
manager = UnifiedExtractionManager(config)

# Process document
result = manager.process_document("invoice.jpg")

print(f"Document Type: {result.document_type}")
print(f"Confidence: {result.confidence_score}")
print(f"Extracted Fields: {result.extracted_fields}")
```

#### Advanced Configuration
```python
from vision_processor.config.unified_config import UnifiedConfig, ModelType
from vision_processor.evaluation.unified_evaluator import UnifiedEvaluator

# Advanced configuration
config = UnifiedConfig()
config.model_type = ModelType.LLAMA32_VISION
config.processing_pipeline = "7step"
config.extraction_method = "hybrid"
config.enable_highlight_detection = True
config.enable_awk_fallback = True
config.confidence_components = 4

# Cross-model evaluation
evaluator = UnifiedEvaluator(config)
results = evaluator.evaluate_dataset(
    dataset_path="test_dataset/",
    model_names=["internvl3", "llama32_vision"],
    output_dir="evaluation_results/"
)
```

## 🧪 Testing

### Comprehensive Test Suite

The project includes extensive testing across multiple levels:

```bash
# Run all tests
pytest tests/ -v

# Unit tests
pytest tests/unit/ -v

# Integration tests  
pytest tests/integration/ -v

# Performance tests
pytest tests/performance/ -v

# Phase validation tests
python test_phase1_foundation.py
python test_phase2_models.py
python test_phase3_pipeline.py
python test_phase4_features.py
python test_phase5_handlers.py
python test_phase6_evaluation.py
python test_phase7_cli.py
python test_phase8_comprehensive.py
```

### Code Quality

```bash
# Linting and formatting
ruff check . --fix
ruff format .

# Type checking
mypy vision_processor/

# Coverage
pytest tests/ --cov=vision_processor --cov-report=html
```

## 📊 Supported Document Types

The system supports 11 Australian tax document types with specialized handlers:

1. **Fuel Receipts** - Petrol station transactions
2. **Tax Invoices** - GST-compliant invoices  
3. **Business Receipts** - General business expenses
4. **Bank Statements** - Financial transaction records
5. **Meal Receipts** - Restaurant and food expenses
6. **Accommodation** - Hotel and lodging receipts
7. **Travel Documents** - Transportation expenses
8. **Parking/Toll** - Vehicle-related fees
9. **Professional Services** - Consultant and service fees
10. **Equipment/Supplies** - Office and business supplies
11. **Other Documents** - Miscellaneous tax-related documents

## 📈 Evaluation Framework

### SROIE Dataset Support
- Complete SROIE dataset evaluation
- Standardized metrics calculation
- Cross-model performance comparison
- Automated report generation

### Evaluation Metrics
- **Extraction Accuracy**: Field-level precision and recall
- **Confidence Calibration**: Confidence vs accuracy correlation
- **Processing Time**: Performance benchmarking
- **ATO Compliance**: Australian tax validation rates
- **Production Readiness**: 5-level assessment score

### Model Comparison
- InternVL3 vs Llama-3.2-Vision comparative analysis
- Statistical significance testing
- Fairness validation across document types
- Hardware performance optimization

## 🏭 Production Deployment

### Hardware Requirements

**Development Environment:**
- 2x H200 GPU system
- Multi-GPU processing enabled
- High VRAM for model optimization

**Production Environment:**
- Single V100 GPU (16GB VRAM)
- 64GB RAM
- CUDA 11.x compatibility

### Optimization Features
- 8-bit quantization for memory efficiency
- Gradient checkpointing for large models
- Device-specific optimizations (CUDA, MPS, CPU)
- Graceful scaling from multi-GPU to single GPU

### Offline Mode
- Complete offline operation support
- Local model loading without internet
- Pre-downloaded model weights
- Production-safe configuration

## 📝 Documentation

### API Documentation
All classes and methods include comprehensive docstrings with:
- Parameter descriptions and types
- Return value specifications
- Usage examples
- Exception handling details

### Configuration Guide
- Environment variable reference
- Hardware optimization settings
- Model-specific configurations
- Production deployment checklist

## 🤝 Contributing

### Development Workflow
1. Create feature branch from main
2. Implement changes with comprehensive tests
3. Run full test suite and linting
4. Submit pull request with detailed description

### Code Standards
- Python 3.11+ features and type hints
- Maximum line length: 108 characters
- Comprehensive test coverage (>80%)
- Google-style docstrings

### Testing Requirements
- Unit tests for all new functionality
- Integration tests for pipeline components
- Performance tests for optimization features
- Documentation updates for API changes

## 📄 License

This project is proprietary software. All rights reserved.

## 🙋 Support

For technical support, feature requests, or bug reports, please contact the development team or create an issue in the project repository.

---

**Built with ❤️ for Australian tax document processing**