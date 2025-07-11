# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment

### Hardware Setup
- **Local Development**: Mac M1 for code editing and planning
- **Remote Development**: 2x H200 GPU system for training and testing
- **Production Target**: Single V100 GPU (16GB VRAM) with 64GB RAM
- **Code Synchronization**: Git for syncing between Mac and development machine

### GPU Memory Constraints
- Development system supports multi-GPU with high VRAM
- Production system requires optimization for single V100 (16GB VRAM)
- Use memory-efficient techniques: 8-bit quantization, gradient checkpointing
- Design models to gracefully scale from multi-GPU to single GPU deployment

## Project Overview

This is the **Unified Vision Document Processing Architecture** repository - a comprehensive system design for consolidating InternVL PoC and Llama-3.2 vision implementations into a single, model-agnostic document processing package focused on Australian tax documents.

### Architecture Foundation
- **Base Architecture**: Llama-3.2's 7-step processing pipeline (Classification → Primary Extraction → AWK Fallback → Validation → ATO Compliance → Confidence Scoring → Recommendations)
- **Technical Integration**: InternVL's advanced capabilities (multi-GPU optimization, computer vision, cross-platform configuration)
- **Processing Approach**: Graceful degradation with multi-tier fallbacks
- **Domain Focus**: Australian Tax Office (ATO) compliance and document processing

## Key Design Principles

1. **Model Agnostic**: Business logic completely independent of vision model choice (InternVL3 or Llama-3.2-Vision)
2. **Llama Architecture Foundation**: 7-step processing pipeline as unified standard
3. **Production Ready**: 5-level production readiness assessment with automated decisions
4. **Australian Tax Focused**: Complete ATO compliance with unified domain expertise
5. **Technical Excellence**: Multi-GPU optimization, highlight detection, enhanced parsing

## Package Structure Overview

The planned architecture follows this structure:
```
unified_vision_processor/
├── vision_processor/                 # Main package
│   ├── config/                      # Unified configuration with model factory
│   ├── models/                      # Model abstraction (InternVL3 + Llama-3.2)
│   ├── classification/              # Document classification (11 Australian tax types)
│   ├── extraction/                  # 7-step Llama pipeline + InternVL enhancements
│   ├── confidence/                  # 4-component confidence scoring
│   ├── compliance/                  # ATO compliance validation
│   ├── handlers/                    # Document-specific processors (11 types)
│   ├── computer_vision/             # InternVL highlight detection
│   ├── evaluation/                  # Cross-model evaluation framework
│   ├── prompts/                     # 60+ specialized prompts
│   ├── cli/                         # Unified command interfaces
│   ├── banking/                     # Australian banking integration
│   └── utils/                       # Shared utilities
```

## Package Dependencies

### environment.yml
```yaml
name: unified_vision_processor
channels:
  - conda-forge
  - defaults
variables:
  KMP_DUPLICATE_LIB_OK: "TRUE"
dependencies:
  - python=3.11
  - numpy
  - pandas
  - pillow
  - matplotlib
  - tqdm
  - pyyaml
  - scikit-learn
  - pip
  - ipykernel  # Required for Jupyter notebook support
  - ipywidgets  # Required for tqdm progress bars in Jupyter
  - pip:
    # Core dependencies for vision processing
    - transformers==4.45.2  # Fixed version for Llama-3.2-Vision compatibility
    - typer>=0.9.0
    - rich>=13.0.0
    - torch>=2.0.0
    - torchvision
    - accelerate  # Required for MPS/device mapping support
    - bitsandbytes  # Required for 8-bit quantization on V100 16GB
    - sentencepiece  # Required for tokenizer
    - protobuf  # Required for model loading
    - python-dotenv  # Required for .env file loading
```

## Configuration Management

The system uses environment-driven configuration with these key settings:

```bash
# Model Selection
VISION_MODEL_TYPE=internvl3          # internvl3 | llama32_vision
VISION_PROCESSING_PIPELINE=7step     # Llama 7-step pipeline standard
VISION_EXTRACTION_METHOD=hybrid      # hybrid | key_value | awk_only

# Feature Integration
VISION_HIGHLIGHT_DETECTION=true      # InternVL highlight detection
VISION_AWK_FALLBACK=true            # Comprehensive AWK fallback
VISION_GRACEFUL_DEGRADATION=true    # Llama graceful degradation
VISION_CONFIDENCE_COMPONENTS=4       # 4-component confidence scoring

# GPU Configuration
VISION_GPU_MEMORY_LIMIT=15360        # V100 16GB limit with buffer (MB)
VISION_ENABLE_8BIT_QUANTIZATION=true # Memory optimization for production
VISION_MULTI_GPU_DEV=true           # Enable for 2x H200 development
VISION_SINGLE_GPU_PROD=true         # Target single V100 production
```

## Development Commands

Since this is currently a planning repository, these commands are for future implementation:

### Environment Setup
```bash
# Create unified conda environment
conda env create -f environment.yml
conda activate unified_vision_processor

# Install PyTorch with appropriate CUDA support
# For V100 production (CUDA 11.x):
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# For H200 development (CUDA 12.x):
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Register as Jupyter kernel
python -m ipykernel install --user --name unified_vision_processor --display-name "Python (unified_vision_processor)"

# Install package in development mode
pip install -e .

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import bitsandbytes; print('BitsAndBytes: OK')"
```

### Quality Assurance (when code exists)
```bash
# Code linting and formatting (pre-commit hook will run automatically)
ruff check . --fix
ruff format .

# Type checking
mypy vision_processor/

# Testing
pytest tests/ -v --cov=vision_processor
pytest tests/integration/ -v
pytest tests/evaluation/ -v
```

### Git Workflow
```bash
# Pre-commit hook automatically runs ruff check before commits
# Commits will fail if ruff check fails - fix issues before committing

# Standard git workflow
git add .
git commit -m "feat: implement document classification system"
git push origin main

# Sync between Mac and development machine
git pull origin main  # On development machine
git push origin main  # After development work
```

### Commit Message Guidelines
- Use conventional commit format: `type: description`
- Common types: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`
- **NEVER include Claude attributions in commit messages**
- You are the developer - commit messages should reflect your work

### CLI Usage (planned)
```bash
# Single document processing
python -m vision_processor.cli single_document image.jpg --model internvl3

# Batch processing
python -m vision_processor.cli batch_processing datasets/ --output results/

# Model comparison
python -m vision_processor.cli model_comparison datasets/ --models internvl3,llama32_vision

# Evaluation
python -m vision_processor.cli evaluation datasets/ ground_truth/ --metric sroie
```

## Implementation Phases

The architecture document outlines an 8-phase, 16-week implementation plan:

1. **Foundation Setup** (Weeks 1-2): Package structure, BaseVisionModel abstraction
2. **Model Integration** (Weeks 3-4): InternVL + Llama model migration
3. **Llama Pipeline Integration** (Weeks 5-6): 7-step processing pipeline
4. **Feature Integration** (Weeks 7-8): Computer vision, AWK systems, ATO compliance
5. **Handler and Prompt Integration** (Weeks 9-10): Document handlers, prompt library
6. **Evaluation Framework** (Weeks 11-12): Unified evaluation system
7. **CLI and Production Features** (Weeks 13-14): Command interfaces, monitoring
8. **Testing and Validation** (Weeks 15-16): Comprehensive testing

## Australian Tax Document Processing

The system is designed to process 11 Australian tax document types with specialized handlers:
- Fuel receipts, Tax invoices, Business receipts, Bank statements
- Meal receipts, Accommodation, Travel documents, Parking/toll
- Professional services, Equipment/supplies, Other documents

### ATO Compliance Features
- ABN validation and format checking
- GST calculation verification (10% standard rate)
- Australian business name recognition (100+ businesses)
- Date format validation (DD/MM/YYYY)
- Australian banking integration (BSB validation)

## Technical Integrations

### InternVL Advanced Features
- Multi-GPU auto-configuration with 8-bit quantization
- Computer vision highlight detection for bank statements
- Enhanced key-value parsing with AWK fallback
- Cross-platform deployment (Mac M1 ↔ 2x H200 ↔ single V100)

### Llama Processing Excellence  
- 7-step processing pipeline with graceful degradation
- 4-component confidence scoring system
- 5-level production readiness assessment
- Comprehensive AWK extraction rules (2,000+ rules)

## Testing Strategy

When implementing, ensure comprehensive testing:
- **Unit tests**: All vision_processor components
- **Integration tests**: End-to-end pipeline processing
- **Evaluation tests**: SROIE dataset validation
- **Parity validation**: Ensure extraction consistency
- **Performance tests**: Cross-model benchmarking

## Migration and Legacy Support

The unified system provides migration tools for existing InternVL and Llama-3.2 implementations, ensuring backward compatibility while providing enhanced capabilities through the unified architecture.

## The Orininal Files Needed for Implementation are here
- [InternVL PoC](/Users/tod/Desktop/internvl_PoC/internvl_git)
- [Llama-3.2 Vision](/Users/tod/Desktop/Llama_3.2/llama_vision)

## Local Testing and Development
- Use the unified_vision_processor conda environment for local development

## Important Notes
- always run `ruff check . --fix --ignore ARG001,ARG002,F841` to ensure code quality
- to run code, you must prefix with "export KMP_DUPLICATE_LIB_OK=TRUE && source /opt/homebrew/Caskroom/miniforge/base/bin/activate unified_vision_processor"