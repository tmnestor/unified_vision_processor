# Unified Vision Document Processing Architecture

**Single Source of Truth Document - Master Implementation Plan**

## Executive Summary

This document defines the unified architecture for vision-based document processing that consolidates the InternVL PoC and Llama-3.2 implementations into a single, model-agnostic package. **The Llama vision approach has been selected as the architectural foundation** due to its sophisticated 7-step processing pipeline, graceful degradation capabilities, and production-ready design. The unified system integrates InternVL's advanced technical capabilities (multi-GPU optimization, computer vision, cross-platform configuration) into the Llama architectural framework.

## Current State Analysis - Both Systems Enhanced

### InternVL PoC System (Current State After Parity Updates)

**Current Capabilities:**
- **Enhanced Key-Value Parser**: Original KEY-VALUE extraction with AWK fallback when <4 fields extracted
- **AWK Extraction System**: Document-specific extractors (Fuel, Tax Invoice, Bank Statement, Other)
- **11 Document Types**: Automatic classification with >0.8 confidence threshold (fail-fast approach)
- **Australian Tax Domain**: ATO compliance assessment, ABN validation, GST verification
- **Bank Statement Processing**: Advanced highlight detection with OCR integration
- **47 Specialized Prompts**: Document-specific extraction optimization
- **Multi-GPU Auto-Configuration**: Intelligent device management with 8-bit quantization
- **Cross-Platform Architecture**: Environment-driven configuration for Mac M1 ↔ multi-GPU deployment
- **Comprehensive Test Suite**: Parity validation tests ensuring extraction consistency

### Llama-3.2 System (Current State After Phase 2A Completion)

**Current Capabilities:**
- **7-Step Processing Pipeline**: Classification → Primary Extraction → AWK Fallback → Validation → ATO Compliance → Confidence Scoring → Recommendations
- **11 Document-Specific Handlers**: Specialized handlers extending BaseATOHandler for each document type
- **Hybrid Extraction Manager**: Multi-tiered extraction with automatic handler routing and batch processing
- **Advanced Confidence Integration**: 4-component confidence scoring with 5-level production readiness assessment
- **Comprehensive AWK System**: 2,000+ line extraction rules with document-specific logic
- **ATO Compliance Framework**: Complete Australian tax validation with business name recognition (100+ businesses)
- **Production-Ready CLI**: Single and batch processing with performance monitoring
- **Australian Tax Prompts**: 13 specialized ATO-compliant prompts optimized for document types

## Architectural Comparison - Current State

### Processing Approaches

| Aspect | InternVL PoC (Enhanced) | Llama-3.2 (Phase 2A) |
|--------|------------------------|----------------------|
| **Processing Flow** | Classification → Type-Specific Processor → AWK Fallback | 7-Step Pipeline: Classification → Handler → Primary → AWK → Validation → ATO → Confidence |
| **Error Handling** | Fail-fast (≥0.8 confidence required) | Graceful degradation with multi-tier fallbacks |
| **Extraction Strategy** | Enhanced KEY-VALUE with automatic AWK fallback | Primary + AWK with quality-based switching |
| **Confidence Scoring** | Classification-based with ATO validation | 4-component weighted scoring (Classification 25%, Extraction 35%, ATO 25%, Business 15%) |
| **Production Readiness** | Binary (pass/fail based on confidence) | 5-level assessment (Excellent/Good/Fair/Poor/Very Poor) |
| **Document Handlers** | Type-specific processors | Specialized handlers with inheritance hierarchy |
| **Bank Processing** | Advanced computer vision with highlight detection | Transaction categorization with work expense scoring |

### Shared Capabilities (Architectural Parity)

**Both Systems Now Have:**
- ✅ **11 Australian Tax Document Types**: Same document taxonomy and classification
- ✅ **ATO Compliance Validation**: ABN format, GST calculation (10%), date format (DD/MM/YYYY)
- ✅ **AWK Fallback Extraction**: Automated fallback when primary extraction insufficient
- ✅ **Australian Business Recognition**: Specialized knowledge of Australian business names and formats
- ✅ **Environment-Driven Configuration**: Cross-platform deployment via environment variables
- ✅ **Batch Processing Capabilities**: Scalable document processing for production use
- ✅ **Comprehensive Testing**: Validation of extraction accuracy and consistency

## Unified Architecture Design

### Core Design Principles

1. **Llama Architecture Foundation**: Use Llama's 7-step processing pipeline as the unified standard
2. **Model Agnostic**: Business logic completely independent of vision model choice
3. **Graceful Degradation**: Multi-tier processing with intelligent fallbacks
4. **Production Ready**: 5-level production readiness assessment with automated decisions
5. **Technical Excellence**: Integrate InternVL's advanced technical capabilities
6. **Australian Tax Focused**: Complete ATO compliance with unified domain expertise
7. **Extensible**: Easy addition of new models and document types

### Unified Package Structure

```
unified_vision_processor/
├── README.md                           # Comprehensive documentation
├── environment.yml                     # Unified conda environment
├── .env.example                        # Configuration template
├── prompts.yaml                        # Combined prompt library (47+ InternVL + 13 Llama prompts)
├── setup.py                           # Package installation
├── requirements.txt                    # Python dependencies
│
├── vision_processor/                   # Main unified package
│   ├── __init__.py
│   │
│   ├── config/                        # Unified configuration management
│   │   ├── __init__.py
│   │   ├── unified_config.py          # Combined configuration with strategy selection
│   │   ├── model_factory.py           # Model selection (InternVL3 | Llama-3.2-Vision)
│   │   ├── processing_strategy.py     # Processing approach selection (fail-fast | graceful)
│   │   ├── prompt_manager.py          # Unified prompt system (60+ prompts)
│   │   └── device_manager.py          # Multi-GPU auto-configuration
│   │
│   ├── models/                        # Model abstraction layer
│   │   ├── __init__.py
│   │   ├── base_model.py              # Abstract base class with standardized response
│   │   ├── internvl_model.py          # InternVL3 with multi-GPU and quantization
│   │   ├── llama_model.py             # Llama-3.2-Vision implementation
│   │   └── model_utils.py             # Device optimization and memory management
│   │
│   ├── classification/                # Unified document classification
│   │   ├── __init__.py
│   │   ├── document_classifier.py     # Combined classification logic from both systems
│   │   ├── confidence_validator.py    # Configurable confidence thresholds
│   │   ├── australian_tax_types.py    # 11 document types with unified taxonomy
│   │   └── classification_exceptions.py # Structured exception handling
│   │
│   ├── extraction/                    # Unified extraction pipeline (Llama-based)
│   │   ├── __init__.py
│   │   ├── hybrid_extraction_manager.py     # Llama 7-step pipeline (foundation)
│   │   ├── enhanced_key_value_parser.py     # InternVL parser integrated into pipeline
│   │   ├── awk_extractor.py                 # Combined AWK systems (2,000+ rules)
│   │   ├── extraction_validator.py          # Multi-tier validation
│   │   └── schema_converter.py              # Format standardization
│   │
│   ├── confidence/                    # Advanced confidence scoring (Llama-based)
│   │   ├── __init__.py
│   │   ├── confidence_integration_manager.py # 4-component confidence scoring (foundation)
│   │   ├── australian_tax_confidence_scorer.py # Advanced confidence assessment
│   │   ├── production_assessor.py     # 5-level production readiness
│   │   └── quality_monitor.py         # Real-time quality control
│   │
│   ├── compliance/                    # Unified ATO compliance
│   │   ├── __init__.py
│   │   ├── ato_compliance_unified.py  # Combined ATO validation from both systems
│   │   ├── australian_tax_validator.py # Business rules and thresholds
│   │   ├── gst_validator.py           # 10% GST validation with tolerance
│   │   ├── business_recognizer.py     # 100+ Australian business patterns
│   │   └── field_validator.py         # Document-specific validation
│   │
│   ├── handlers/                      # Document-specific processing (Llama-based)
│   │   ├── __init__.py
│   │   ├── base_ato_handler.py        # Abstract handler with 7-step pipeline (foundation)
│   │   ├── fuel_receipt_handler.py    # Vehicle expense validation
│   │   ├── tax_invoice_handler.py     # GST compliance and ABN validation
│   │   ├── business_receipt_handler.py # Item-level extraction
│   │   ├── bank_statement_handler.py   # Transaction categorization + highlight detection
│   │   ├── meal_receipt_handler.py     # Entertainment expense validation
│   │   ├── accommodation_handler.py    # Travel expense validation
│   │   ├── travel_document_handler.py  # Travel documentation
│   │   ├── parking_toll_handler.py     # Parking/toll validation
│   │   ├── professional_services_handler.py # Legal/accounting services
│   │   ├── equipment_supplies_handler.py    # Equipment/supplies validation
│   │   └── other_document_handler.py   # Fallback processing
│   │
│   ├── computer_vision/               # Advanced image processing (InternVL features)
│   │   ├── __init__.py
│   │   ├── highlight_detector.py      # Multi-color highlight detection
│   │   ├── ocr_processor.py           # OCR from highlighted regions
│   │   ├── bank_statement_cv.py       # Bank statement computer vision
│   │   ├── image_preprocessor.py      # Image optimization and validation
│   │   └── spatial_correlator.py      # Highlight-text correlation
│   │
│   ├── evaluation/                    # Unified evaluation framework
│   │   ├── __init__.py
│   │   ├── unified_evaluator.py       # Cross-model evaluation with Llama pipeline
│   │   ├── sroie_evaluator.py         # SROIE dataset evaluation (enhanced)
│   │   ├── model_comparator.py        # Fair model comparison framework
│   │   ├── metrics_calculator.py      # Advanced metrics computation
│   │   └── report_generator.py        # Comprehensive evaluation reporting
│   │
│   ├── prompts/                       # Unified prompt system
│   │   ├── __init__.py
│   │   ├── prompt_factory.py          # Dynamic prompt selection
│   │   ├── internvl_prompts.py        # 47 InternVL specialized prompts
│   │   ├── llama_prompts.py           # 13 Llama ATO prompts
│   │   ├── prompt_optimizer.py        # Prompt performance optimization
│   │   └── prompt_validator.py        # Prompt compatibility validation
│   │
│   ├── cli/                           # Unified command-line interfaces
│   │   ├── __init__.py
│   │   ├── unified_cli.py             # Main CLI with Llama pipeline
│   │   ├── single_document.py         # Single document processing
│   │   ├── batch_processing.py        # Batch processing with statistics
│   │   ├── model_comparison.py        # Cross-model comparison
│   │   └── evaluation_cli.py          # Evaluation interface
│   │
│   ├── banking/                       # Australian banking integration
│   │   ├── __init__.py
│   │   ├── bank_recognizer.py         # 11 major Australian banks
│   │   ├── bsb_validator.py           # Bank State Branch validation
│   │   ├── transaction_categorizer.py # Work expense categorization
│   │   └── highlight_processor.py     # Bank statement highlight integration
│   │
│   └── utils/                         # Shared utilities
│       ├── __init__.py
│       ├── logging_config.py          # Unified logging system
│       ├── path_manager.py            # Cross-platform path management
│       ├── performance_monitor.py     # Performance tracking
│       ├── error_handler.py           # Comprehensive error handling
│       └── migration_tools.py         # Tools for migrating existing data
│
├── datasets/                          # Unified datasets (flat structure)
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
├── ground_truth/                      # Evaluation ground truth
│   ├── image1.json
│   ├── image2.json
│   └── ...
│
├── output/                           # Processing results
│   ├── predictions/
│   ├── evaluations/
│   ├── comparisons/
│   └── strategy_analysis/
│
├── tests/                            # Comprehensive test suite
│   ├── unit/
│   ├── integration/
│   ├── evaluation/
│   ├── parity_validation/
│   └── performance/
│
└── docs/                             # Documentation
    ├── api/
    ├── configuration/
    ├── migration_guide/
    └── comparison_framework/
```

## Configuration Management

### Unified Environment Configuration

```bash
# .env.example - Unified Configuration

# Model Selection
VISION_MODEL_TYPE=internvl3              # internvl3 | llama32_vision
VISION_MODEL_PATH=/path/to/model
VISION_DEVICE_CONFIG=auto                # auto | cpu | cuda:0 | multi_gpu

# Processing Configuration (Llama-based)
VISION_PROCESSING_PIPELINE=7step         # Use Llama 7-step pipeline as standard
VISION_EXTRACTION_METHOD=hybrid          # hybrid | key_value | awk_only
VISION_QUALITY_THRESHOLD=0.6             # Quality threshold for processing decisions
VISION_CONFIDENCE_THRESHOLD=0.8          # High confidence threshold for auto-approval

# Feature Integration
VISION_HIGHLIGHT_DETECTION=true          # Enable InternVL highlight detection integration
VISION_AWK_FALLBACK=true                 # Enable comprehensive AWK fallback
VISION_COMPUTER_VISION=true              # Enable InternVL computer vision features
VISION_GRACEFUL_DEGRADATION=true         # Enable Llama graceful degradation

# Confidence and Production (Llama-based)
VISION_CONFIDENCE_COMPONENTS=4           # 4-component confidence scoring
VISION_PRODUCTION_ASSESSMENT=5level      # 5-level production readiness

# Data Paths
VISION_DATASET_PATH=/path/to/datasets
VISION_GROUND_TRUTH_PATH=/path/to/ground_truth
VISION_OUTPUT_PATH=/path/to/output

# Performance and Compatibility
VISION_BATCH_SIZE=1
VISION_MAX_WORKERS=4
VISION_GPU_MEMORY_FRACTION=0.8
VISION_CROSS_PLATFORM=true

# Evaluation and Comparison
VISION_FAIR_COMPARISON=true              # Ensure identical Llama pipeline for both models
VISION_MODEL_COMPARISON=true             # Compare models using identical processing
VISION_EVALUATION_FIELDS=date_value,store_name_value,tax_value,total_value
VISION_SROIE_EVALUATION=true
```

## Unified Processing Pipeline (Llama-Based Foundation)

### Standard 7-Step Processing Architecture

```python
# vision_processor/extraction/hybrid_extraction_manager.py

class UnifiedExtractionManager:
    """Unified manager using Llama 7-step pipeline as foundation."""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.model = ModelFactory.create_model(
            ModelType(config.model_type),
            config.model_path,
            config
        )
        # Llama architecture components
        self.classifier = AustralianTaxClassifier(config)
        self.awk_extractor = AWKExtractor(config)
        self.confidence_manager = ConfidenceIntegrationManager(config)
        self.ato_compliance = ATOComplianceHandler(config)
        
        # InternVL technical integrations
        self.highlight_detector = HighlightDetector(config) if config.highlight_detection else None
        self.enhanced_parser = EnhancedKeyValueParser(config)
        
    def process_document(self, image_path: Path, document_type: Optional[str] = None) -> ProcessingResult:
        """Process document using unified 7-step Llama pipeline with InternVL integrations."""
        
        start_time = time.time()
        
        # Step 1: Document Classification (Llama approach with graceful handling)
        if document_type:
            classified_type = DocumentType(document_type)
            confidence = 1.0
        else:
            classified_type, confidence, evidence = self.classifier.classify_with_evidence(image_path)
            # Graceful degradation: proceed even with lower confidence
            logger.info(f"Classification: {classified_type.value} (confidence: {confidence:.2f})")
        
        # InternVL Integration: Computer Vision Processing
        highlights = []
        if self.highlight_detector and classified_type == DocumentType.BANK_STATEMENT:
            highlights = self.highlight_detector.detect_highlights(image_path)
            logger.info(f"Detected {len(highlights)} highlights")
        
        # Step 2: Model Inference
        prompt = self.prompt_manager.get_prompt(classified_type, has_highlights=bool(highlights))
        model_response = self.model.process_image(image_path, prompt)
        
        # Step 3: Handler Selection and Primary Extraction
        handler = self._get_handler(classified_type)
        extracted_fields = handler.extract_fields_primary(model_response.raw_text)
        
        # InternVL Integration: Enhanced Key-Value Parser
        if self.config.use_enhanced_parser:
            enhanced_fields = self.enhanced_parser.parse(model_response.raw_text)
            extracted_fields = self._merge_extractions(extracted_fields, enhanced_fields)
        
        # Step 4: AWK Fallback (if extraction quality insufficient)
        awk_used = False
        if self._extraction_quality_insufficient(extracted_fields):
            awk_fields = self.awk_extractor.extract(model_response.raw_text, classified_type)
            extracted_fields = self._merge_extractions(extracted_fields, awk_fields)
            awk_used = True
            logger.info("AWK fallback extraction applied")
        
        # Step 5: Field Validation
        validated_fields = handler.validate_fields(extracted_fields)
        
        # InternVL Integration: Highlight Enhancement
        if highlights and classified_type == DocumentType.BANK_STATEMENT:
            validated_fields = self._enhance_with_highlights(validated_fields, highlights)
        
        # Step 6: ATO Compliance Assessment
        compliance_result = self.ato_compliance.assess_compliance(
            validated_fields, 
            classified_type
        )
        
        # Step 7: Confidence Integration and Production Readiness (Llama 4-component system)
        confidence_result = self.confidence_manager.assess_document_confidence(
            model_response.raw_text,
            validated_fields,
            compliance_result,
            classification_confidence=confidence,
            highlights_detected=bool(highlights)
        )
        
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            model_type=self.config.model_type,
            document_type=classified_type.value,
            raw_response=model_response.raw_text,
            extracted_fields=validated_fields,
            awk_fallback_used=awk_used,
            highlights_detected=len(highlights),
            confidence_score=confidence_result.overall_confidence,
            quality_grade=confidence_result.quality_grade,
            ato_compliance_score=compliance_result.compliance_score,
            production_ready=confidence_result.production_ready,
            processing_time=processing_time,
            quality_flags=confidence_result.quality_flags,
            recommendations=confidence_result.recommendations
        )
```

## Migration Implementation Strategy

### Phase 1: Foundation Setup (Week 1-2)
- [ ] **Create unified package structure** based on Llama architecture
- [ ] **Implement BaseVisionModel abstraction** preserving both model capabilities
- [ ] **Create ModelFactory** with InternVL multi-GPU optimization integrated
- [ ] **Implement UnifiedConfig** using Llama configuration with InternVL features
- [ ] **Set up Llama 7-step pipeline framework**

### Phase 2: Model Integration (Week 3-4)
- [ ] **Migrate InternVLModel** with multi-GPU auto-configuration and quantization
- [ ] **Migrate LlamaVisionModel** with current optimization
- [ ] **Test model response standardization** ensuring compatibility
- [ ] **Validate device management** across both models
- [ ] **Performance benchmark** both models in unified system

### Phase 3: Llama Pipeline Integration (Week 5-6)
- [ ] **Implement Llama 7-step processing pipeline** as unified standard
- [ ] **Integrate InternVL enhanced key-value parser** into Step 3
- [ ] **Migrate Llama classification system** with graceful degradation
- [ ] **Implement 4-component confidence scoring** from Llama system
- [ ] **Test unified pipeline** with both models

### Phase 4: Feature Integration (Week 7-8)
- [ ] **Migrate computer vision capabilities** (InternVL highlight detection)
- [ ] **Integrate AWK systems** from both codebases (enhanced + comprehensive)
- [ ] **Combine ATO compliance** validation from both systems
- [ ] **Merge Australian business recognition** (100+ businesses)
- [ ] **Test feature compatibility** and performance

### Phase 5: Handler and Prompt Integration (Week 9-10)
- [ ] **Migrate Llama document handlers** as foundation (11 handlers with 7-step pipeline)
- [ ] **Integrate InternVL prompt library** (47 prompts + 13 Llama = 60+ prompts)
- [ ] **Enhance handlers with InternVL features** (highlight detection, enhanced parsing)
- [ ] **Test handler performance** with unified pipeline
- [ ] **Validate prompt effectiveness** for both models

### Phase 6: Evaluation Framework (Week 11-12)
- [ ] **Create unified evaluation system** using Llama pipeline for fair comparison
- [ ] **Integrate InternVL SROIE evaluation** framework
- [ ] **Create model comparison** tools with identical Llama processing
- [ ] **Implement performance benchmarking** across both models
- [ ] **Test evaluation accuracy** and consistency

### Phase 7: CLI and Production Features (Week 13-14)
- [ ] **Implement unified CLI** using Llama pipeline with model selection
- [ ] **Create model comparison interfaces** with identical processing
- [ ] **Add production monitoring** using Llama 5-level assessment
- [ ] **Implement batch processing** with Llama statistics generation
- [ ] **Test CLI usability** and functionality

### Phase 8: Testing and Validation (Week 15-16)
- [ ] **Comprehensive unit testing** for unified Llama-based components
- [ ] **Model fairness testing** with identical Llama pipeline processing
- [ ] **Performance validation** against original systems
- [ ] **Production readiness testing** using 5-level assessment
- [ ] **InternVL feature integration testing** (highlights, multi-GPU, etc.)

## Success Criteria

### Technical Success Criteria
- [ ] **Model Agnostic Processing**: Both models process documents through identical Llama 7-step pipeline
- [ ] **Llama Architecture Foundation**: Unified system based on Llama's sophisticated processing approach
- [ ] **InternVL Feature Integration**: Advanced technical capabilities seamlessly integrated
- [ ] **Performance Parity**: Processing performance equal to or better than original systems
- [ ] **Graceful Degradation**: Multi-tier processing with intelligent fallbacks

### Business Success Criteria
- [ ] **Fair Model Comparison**: Unbiased comparison using identical Llama pipeline
- [ ] **Production Ready**: 5-level assessment with automated deployment decisions
- [ ] **Reduced Maintenance**: Single codebase with unified Llama-based architecture
- [ ] **Enhanced Capabilities**: Combined strengths exceed individual system capabilities
- [ ] **Unified Processing**: Single sophisticated pipeline for all document types

## Conclusion

The unified architecture uses the Llama-3.2 system's sophisticated 7-step processing pipeline as the foundation, integrating InternVL PoC's advanced technical capabilities (multi-GPU optimization, computer vision, cross-platform configuration) to create a production-ready, model-agnostic document processing system. This approach eliminates architectural complexity while preserving all advanced features from both systems.

**Key Benefits:**
- **Llama Architecture Foundation**: Sophisticated 7-step pipeline with graceful degradation
- **InternVL Technical Excellence**: Multi-GPU optimization, highlight detection, enhanced parsing
- **Fair Model Comparison**: Identical Llama pipeline eliminates architectural bias
- **Production Ready**: 5-level assessment with automated deployment decisions
- **Unified Maintenance**: Single Llama-based codebase with integrated improvements
- **Enhanced Capabilities**: Combined strengths exceed individual system capabilities

---

**Document Status**: Active - Single Source of Truth  
**Last Updated**: Based on Current Enhanced State of Both Systems  
**Implementation Timeline**: 16 weeks (4 months)  
**Systems Analyzed**: InternVL PoC (enhanced with AWK + ATO) + Llama-3.2 (Phase 2A complete)