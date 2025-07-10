# Unified Vision Document Processing Architecture

A comprehensive system for consolidating InternVL PoC and Llama-3.2 vision implementations into a single, model-agnostic document processing package focused on Australian tax documents.

## Phase 1: Foundation Setup ✅

The foundation of the unified architecture has been established with the following components:

### 1. Package Structure (✅ Complete)
- Created comprehensive directory structure following the Llama architecture
- Organized into logical modules: config, models, extraction, classification, etc.
- Prepared for all subsequent phases

### 2. BaseVisionModel Abstraction (✅ Complete)
- Implemented in `vision_processor/models/base_model.py`
- Provides unified interface for both InternVL3 and Llama-3.2-Vision models
- Preserves model-specific optimizations while enabling model-agnostic processing
- Includes standardized response format and device management

### 3. ModelFactory with Multi-GPU Optimization (✅ Complete)
- Implemented in `vision_processor/config/model_factory.py`
- Handles model selection and instantiation
- Integrates InternVL multi-GPU optimization
- Provides device-specific optimizations (CUDA, MPS, CPU)
- Includes recommended configurations for different hardware profiles

### 4. UnifiedConfig Implementation (✅ Complete)
- Implemented in `vision_processor/config/unified_config.py`
- Combines Llama configuration framework with InternVL features
- Environment-driven configuration support
- Auto-detection of hardware environment (Mac M1, H200, V100)
- Comprehensive validation and optimization

### 5. 7-Step Pipeline Framework (✅ Complete)
- Implemented in `vision_processor/extraction/hybrid_extraction_manager.py`
- Established Llama-3.2's 7-step processing pipeline as foundation:
  1. Document Classification
  2. Model Inference
  3. Primary Extraction
  4. AWK Fallback
  5. Field Validation
  6. ATO Compliance
  7. Confidence Integration
- Created placeholder components for future phases
- Integrated InternVL features (highlight detection, enhanced parsing)

## Architecture Overview

The unified system uses the Llama-3.2 7-step processing pipeline as its foundation, integrating InternVL's advanced technical capabilities:

```
Vision Document → Classification → Model Inference → Primary Extraction 
                                                          ↓
                  Confidence ← ATO Compliance ← Validation ← AWK Fallback
```

## Key Features

- **Model Agnostic**: Business logic independent of vision model choice
- **Graceful Degradation**: Multi-tier processing with intelligent fallbacks
- **Australian Tax Focused**: Complete ATO compliance validation
- **Multi-GPU Support**: Optimized for various hardware configurations
- **Production Ready**: 5-level assessment with automated decisions

## Testing

To verify the Phase 1 implementation:

```bash
python test_pipeline.py
```

## Next Steps

Phase 2: Model Integration (Weeks 3-4)
- Migrate InternVLModel implementation
- Migrate LlamaVisionModel implementation
- Test model response standardization
- Validate device management
- Performance benchmarking

## Configuration

The system supports environment-driven configuration. Create a `.env` file:

```bash
# Model Paths - CRITICAL FOR OFFLINE PRODUCTION
VISION_INTERNVL_MODEL_PATH=/Users/tod/PretrainedLLM/InternVL3-8B
VISION_LLAMA_MODEL_PATH=/Users/tod/PretrainedLLM/Llama-3.2-11B-Vision

# Model Selection
VISION_MODEL_TYPE=internvl3              # internvl3 | llama32_vision
VISION_PROCESSING_PIPELINE=7step
VISION_EXTRACTION_METHOD=hybrid

# Features
VISION_HIGHLIGHT_DETECTION=true
VISION_AWK_FALLBACK=true
VISION_GRACEFUL_DEGRADATION=true

# Offline Mode - Default is true for production safety
VISION_OFFLINE_MODE=true                 # Default: true
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

### Production Deployment

For production environments without internet access:

1. Copy pretrained models to local directories
2. Set model paths in `.env` file
3. Enable offline mode flags
4. The system will load models from local paths only

## Development

```bash
# Create conda environment
conda env create -f environment.yml
conda activate unified_vision_processor

# Install in development mode
pip install -e .

# Run tests
python test_pipeline.py
```