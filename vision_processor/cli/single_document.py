"""Single Document Processing CLI

Specialized command-line interface for processing individual documents
with detailed output and debugging capabilities.
"""

import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..classification import DocumentType
from ..config.unified_config import ModelType, UnifiedConfig
from ..extraction.hybrid_extraction_manager import UnifiedExtractionManager

# Initialize app and console
app = typer.Typer(
    name="single-document",
    help="Process individual documents with detailed analysis",
    rich_markup_mode="rich",
)
console = Console()

logger = logging.getLogger(__name__)


@app.command()
def process(
    image_path: str = typer.Argument(..., help="Path to document image"),
    model: str = typer.Option(
        "internvl3",
        "--model",
        "-m",
        help="Vision model: internvl3 or llama32_vision",
    ),
    document_type: str | None = typer.Option(
        None,
        "--type",
        "-t",
        help="Document type (fuel_receipt, tax_invoice, business_receipt, etc.)",
    ),
    output_format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, json, detailed",
    ),
    save_output: str | None = typer.Option(
        None,
        "--save",
        "-s",
        help="Save results to file (JSON format)",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Enable debug mode with detailed pipeline info",
    ),
    confidence_threshold: float = typer.Option(
        0.7,
        "--threshold",
        "-c",
        help="Confidence threshold for production readiness",
    ),
) -> None:
    """Process a single document with comprehensive analysis.

    Provides detailed extraction results, confidence scoring, and
    production readiness assessment using the Llama 7-step pipeline.
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        console.print("[yellow]🐛 Debug mode enabled[/yellow]")

    # Validate inputs
    image_file = Path(image_path)
    if not image_file.exists():
        console.print(f"[red]❌ Image file not found: {image_path}[/red]")
        raise typer.Exit(1)

    try:
        model_type = ModelType(model.lower())
    except ValueError:
        console.print(f"[red]❌ Invalid model: {model}[/red]")
        console.print(f"Available models: {', '.join([m.value for m in ModelType])}")
        raise typer.Exit(1) from None

    # Validate document type if provided
    if document_type:
        try:
            DocumentType(document_type.lower())
        except ValueError:
            console.print(f"[red]❌ Invalid document type: {document_type}[/red]")
            valid_types = [
                dt.value for dt in DocumentType if dt != DocumentType.UNKNOWN
            ]
            console.print(f"Available types: {', '.join(valid_types)}")
            raise typer.Exit(1) from None

    console.print(
        Panel.fit(
            f"[bold blue]🔍 Processing Document[/bold blue]\n"
            f"File: [green]{image_file.name}[/green]\n"
            f"Model: [yellow]{model_type.value}[/yellow]\n"
            f"Type: [cyan]{document_type or 'auto-detect'}[/cyan]",
            title="Document Analysis",
        ),
    )

    try:
        # Create configuration
        config = UnifiedConfig.from_env()
        config.model_type = model_type
        config.confidence_threshold = confidence_threshold

        # Process document with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Add processing task
            task = progress.add_task("Initializing unified pipeline...", total=7)

            with UnifiedExtractionManager(config) as extraction_manager:
                # Step-by-step processing with progress updates
                if debug:
                    progress.update(
                        task,
                        description="Step 1: Document Classification",
                        advance=1,
                    )
                    console.print("[dim]🎯 Classifying document type...[/dim]")

                progress.update(task, description="Step 2: Model Inference", advance=1)
                if debug:
                    console.print("[dim]🧠 Processing with vision model...[/dim]")

                progress.update(
                    task,
                    description="Step 3: Primary Extraction",
                    advance=1,
                )
                if debug:
                    console.print("[dim]📝 Extracting fields...[/dim]")

                progress.update(
                    task,
                    description="Step 4: AWK Fallback Check",
                    advance=1,
                )
                progress.update(task, description="Step 5: Field Validation", advance=1)
                progress.update(
                    task,
                    description="Step 6: ATO Compliance Check",
                    advance=1,
                )
                progress.update(
                    task,
                    description="Step 7: Confidence Integration",
                    advance=1,
                )

                # Process document
                result = extraction_manager.process_document(image_file, document_type)

                progress.update(
                    task,
                    description="✅ Processing complete!",
                    completed=7,
                )

        # Display results based on format
        if output_format == "json":
            _display_json_results(result)
        elif output_format == "detailed":
            _display_detailed_results(result, debug)
        else:  # table format
            _display_table_results(result)

        # Production readiness assessment
        _display_production_assessment(result, confidence_threshold)

        # Save output if requested
        if save_output:
            _save_results(result, save_output)
            console.print(f"[green]💾 Results saved to: {save_output}[/green]")

        # Exit with appropriate code based on production readiness
        if result.production_ready:
            console.print(
                "[green]✅ Document processing successful - Production ready[/green]",
            )
        else:
            console.print(
                "[yellow]⚠️ Document processing completed - Manual review recommended[/yellow]",
            )
            if not debug:
                console.print(
                    "[dim]💡 Use --debug for detailed pipeline information[/dim]",
                )

    except Exception as e:
        console.print(f"[red]❌ Processing failed: {e}[/red]")
        if debug:
            import traceback

            console.print("[red]Stack trace:[/red]")
            console.print(traceback.format_exc())
        raise typer.Exit(1) from None


@app.command()
def analyze(
    image_path: str = typer.Argument(..., help="Path to document image"),
    model: str = typer.Option("internvl3", "--model", "-m", help="Vision model to use"),
    compare_models: bool = typer.Option(
        False,
        "--compare/--no-compare",
        help="Compare results from both models",
    ),
    show_pipeline: bool = typer.Option(
        False,
        "--pipeline/--no-pipeline",
        help="Show detailed pipeline execution",
    ),
) -> None:
    """Analyze document with comprehensive pipeline breakdown.

    Provides detailed analysis of each processing step including
    confidence scores, quality assessment, and recommendations.
    """
    image_file = Path(image_path)
    if not image_file.exists():
        console.print(f"[red]❌ Image file not found: {image_path}[/red]")
        raise typer.Exit(1)

    console.print(
        Panel.fit(
            f"[bold blue]🔬 Document Analysis[/bold blue]\n"
            f"File: [green]{image_file.name}[/green]\n"
            f"Analysis Mode: [yellow]{'Comparison' if compare_models else 'Single Model'}[/yellow]",
            title="Advanced Analysis",
        ),
    )

    try:
        if compare_models:
            _analyze_with_comparison(image_file, show_pipeline)
        else:
            _analyze_single_model(image_file, model, show_pipeline)

    except Exception as e:
        console.print(f"[red]❌ Analysis failed: {e}[/red]")
        raise typer.Exit(1) from None


def _display_table_results(result) -> None:
    """Display results in table format."""
    # Main results table
    main_table = Table(title="📋 Processing Results", show_header=True)
    main_table.add_column("Metric", style="cyan", width=20)
    main_table.add_column("Value", style="green")
    main_table.add_column("Status", justify="center", width=10)

    # Add processing information
    main_table.add_row("Model", result.model_type, "🤖")
    main_table.add_row(
        "Document Type",
        result.document_type.replace("_", " ").title(),
        "📄",
    )
    main_table.add_row("Processing Time", f"{result.processing_time:.2f}s", "⏱️")

    # Add quality metrics
    confidence_status = (
        "🟢"
        if result.confidence_score >= 0.8
        else "🟡"
        if result.confidence_score >= 0.6
        else "🔴"
    )
    main_table.add_row(
        "Confidence Score",
        f"{result.confidence_score:.3f}",
        confidence_status,
    )

    quality_emoji = {
        "excellent": "🟢",
        "good": "🟡",
        "fair": "🟠",
        "poor": "🔴",
        "very_poor": "⚫",
    }
    main_table.add_row(
        "Quality Grade",
        result.quality_grade.value.title(),
        quality_emoji.get(result.quality_grade.value, "❓"),
    )

    main_table.add_row(
        "Production Ready",
        "✅ Yes" if result.production_ready else "❌ No",
        "🏭",
    )
    main_table.add_row("ATO Compliance", f"{result.ato_compliance_score:.3f}", "🇦🇺")

    # Add pipeline information
    main_table.add_row(
        "AWK Fallback",
        "Used" if result.awk_fallback_used else "Not Used",
        "🔄" if result.awk_fallback_used else "✅",
    )
    main_table.add_row("Highlights", str(result.highlights_detected), "🔍")

    console.print(main_table)

    # Extracted fields table
    if result.extracted_fields:
        fields_table = Table(title="📝 Extracted Fields", show_header=True)
        fields_table.add_column("Field", style="cyan")
        fields_table.add_column("Value", style="yellow")
        fields_table.add_column("Type", style="dim")

        for field, value in result.extracted_fields.items():
            if not field.startswith("_") and value:
                field_display = field.replace("_", " ").title()
                value_str = str(value)

                # Determine field type
                if "amount" in field.lower() or "total" in field.lower():
                    field_type = "💰 Amount"
                elif "date" in field.lower():
                    field_type = "📅 Date"
                elif "name" in field.lower() or "supplier" in field.lower():
                    field_type = "🏢 Business"
                elif "abn" in field.lower():
                    field_type = "🆔 ABN"
                else:
                    field_type = "📄 Text"

                fields_table.add_row(field_display, value_str, field_type)

        console.print(fields_table)


def _display_json_results(result) -> None:
    """Display results in JSON format."""
    result_dict = {
        "model_type": result.model_type,
        "document_type": result.document_type,
        "processing_time": result.processing_time,
        "confidence_score": result.confidence_score,
        "quality_grade": result.quality_grade.value,
        "production_ready": result.production_ready,
        "ato_compliance_score": result.ato_compliance_score,
        "validation_passed": result.validation_passed,
        "awk_fallback_used": result.awk_fallback_used,
        "highlights_detected": result.highlights_detected,
        "extracted_fields": result.extracted_fields,
        "quality_flags": result.quality_flags,
        "recommendations": result.recommendations,
        "stages_completed": [stage.value for stage in result.stages_completed],
    }

    console.print(JSON.from_data(result_dict))


def _display_detailed_results(result, debug: bool = False) -> None:
    """Display detailed results with pipeline information."""
    # Processing overview
    overview_table = Table(title="🔍 Processing Overview")
    overview_table.add_column("Component", style="cyan")
    overview_table.add_column("Result", style="green")
    overview_table.add_column("Details", style="dim")

    overview_table.add_row(
        "Model Engine",
        result.model_type,
        "Unified pipeline processing",
    )
    overview_table.add_row(
        "Document Classification",
        result.document_type.replace("_", " ").title(),
        f"Confidence: {result.confidence_score:.3f}",
    )
    overview_table.add_row(
        "Quality Assessment",
        result.quality_grade.value.title(),
        f"Production ready: {'Yes' if result.production_ready else 'No'}",
    )
    overview_table.add_row(
        "Processing Performance",
        f"{result.processing_time:.2f}s",
        f"Memory usage: {result.memory_usage_mb:.1f} MB",
    )

    console.print(overview_table)

    # Pipeline stages
    if debug and result.stages_completed:
        stages_table = Table(title="🔧 Pipeline Execution")
        stages_table.add_column("Stage", style="cyan")
        stages_table.add_column("Status", justify="center")
        stages_table.add_column("Description", style="dim")

        stage_descriptions = {
            "classification": "Document type identification",
            "inference": "Vision model processing",
            "primary_extraction": "Field extraction from model output",
            "awk_fallback": "Fallback pattern matching",
            "validation": "Field validation and cleaning",
            "ato_compliance": "Australian tax compliance check",
            "confidence_integration": "Quality assessment and scoring",
        }

        for stage in result.stages_completed:
            stage_name = stage.value.replace("_", " ").title()
            description = stage_descriptions.get(stage.value, "Pipeline processing")
            stages_table.add_row(stage_name, "✅", description)

        console.print(stages_table)

    # Quality analysis
    if result.quality_flags or result.recommendations:
        quality_panel_content = ""

        if result.quality_flags:
            quality_panel_content += "[yellow]Quality Flags:[/yellow]\n"
            for flag in result.quality_flags:
                quality_panel_content += f"  • {flag.replace('_', ' ').title()}\n"

        if result.recommendations:
            quality_panel_content += "\n[blue]Recommendations:[/blue]\n"
            for rec in result.recommendations:
                quality_panel_content += f"  • {rec}\n"

        console.print(Panel(quality_panel_content.strip(), title="🎯 Quality Analysis"))


def _display_production_assessment(result, threshold: float) -> None:
    """Display production readiness assessment."""
    # Determine production status
    if result.production_ready and result.confidence_score >= threshold:
        status_color = "green"
        status_icon = "✅"
        status_text = "READY FOR PRODUCTION"
    elif result.confidence_score >= threshold * 0.8:
        status_color = "yellow"
        status_icon = "⚠️"
        status_text = "REQUIRES MONITORING"
    else:
        status_color = "red"
        status_icon = "❌"
        status_text = "NOT PRODUCTION READY"

    # Create production assessment panel
    assessment_content = f"""[bold {status_color}]{status_icon} {status_text}[/bold {status_color}]

[bold]Confidence Score:[/bold] {result.confidence_score:.3f} (threshold: {threshold:.1f})
[bold]Quality Grade:[/bold] {result.quality_grade.value.title()}
[bold]ATO Compliance:[/bold] {result.ato_compliance_score:.3f}
[bold]Validation Status:[/bold] {"Passed" if result.validation_passed else "Failed"}

[bold]Pipeline Efficiency:[/bold]
• Processing Time: {result.processing_time:.2f}s
• AWK Fallback: {"Used" if result.awk_fallback_used else "Not needed"}
• Highlights Detected: {result.highlights_detected}
• Memory Usage: {result.memory_usage_mb:.1f} MB"""

    console.print(Panel(assessment_content, title="🏭 Production Readiness Assessment"))


def _analyze_single_model(image_file: Path, model: str, show_pipeline: bool) -> None:
    """Analyze document with single model."""
    config = UnifiedConfig.from_env()
    config.model_type = ModelType(model.lower())

    with UnifiedExtractionManager(config) as extraction_manager:
        result = extraction_manager.process_document(image_file)

    _display_detailed_results(result, show_pipeline)
    _display_production_assessment(result, config.confidence_threshold)


def _analyze_with_comparison(image_file: Path, show_pipeline: bool) -> None:
    """Analyze document with model comparison."""
    models = ["internvl3", "llama32_vision"]
    results = {}

    console.print("[bold blue]🔄 Processing with both models...[/bold blue]")

    for model in models:
        with console.status(f"Processing with {model}..."):
            config = UnifiedConfig.from_env()
            config.model_type = ModelType(model)

            with UnifiedExtractionManager(config) as extraction_manager:
                result = extraction_manager.process_document(image_file)
                results[model] = result

    # Comparison table
    comparison_table = Table(title="⚖️ Model Comparison")
    comparison_table.add_column("Metric", style="cyan")
    comparison_table.add_column("InternVL3", style="green")
    comparison_table.add_column("Llama-3.2", style="yellow")
    comparison_table.add_column("Winner", justify="center")

    # Compare key metrics
    metrics = [
        ("Confidence Score", "confidence_score"),
        ("Processing Time", "processing_time"),
        ("ATO Compliance", "ato_compliance_score"),
        ("Production Ready", "production_ready"),
        ("Quality Grade", "quality_grade"),
    ]

    for metric_name, metric_attr in metrics:
        internvl_val = getattr(results["internvl3"], metric_attr)
        llama_val = getattr(results["llama32_vision"], metric_attr)

        if metric_attr == "quality_grade":
            internvl_str = internvl_val.value
            llama_str = llama_val.value
            # Quality comparison based on order
            quality_order = ["very_poor", "poor", "fair", "good", "excellent"]
            winner = (
                "🟢"
                if quality_order.index(internvl_val.value)
                > quality_order.index(llama_val.value)
                else "🟡"
            )
        elif metric_attr == "processing_time":
            internvl_str = f"{internvl_val:.2f}s"
            llama_str = f"{llama_val:.2f}s"
            winner = "🟢" if internvl_val < llama_val else "🟡"  # Lower is better
        elif isinstance(internvl_val, bool):
            internvl_str = "Yes" if internvl_val else "No"
            llama_str = "Yes" if llama_val else "No"
            winner = (
                "🟢"
                if internvl_val and not llama_val
                else "🟡"
                if llama_val and not internvl_val
                else "🤝"
            )
        else:
            internvl_str = f"{internvl_val:.3f}"
            llama_str = f"{llama_val:.3f}"
            winner = "🟢" if internvl_val > llama_val else "🟡"

        comparison_table.add_row(metric_name, internvl_str, llama_str, winner)

    console.print(comparison_table)

    # Show detailed results for both if requested
    if show_pipeline:
        for model_name, result in results.items():
            console.print(f"\n[bold]📊 Detailed Results - {model_name.upper()}[/bold]")
            _display_detailed_results(result, True)


def _save_results(result, output_path: str) -> None:
    """Save processing results to JSON file."""
    result_dict = {
        "model_type": result.model_type,
        "document_type": result.document_type,
        "processing_time": result.processing_time,
        "confidence_score": result.confidence_score,
        "quality_grade": result.quality_grade.value,
        "production_ready": result.production_ready,
        "ato_compliance_score": result.ato_compliance_score,
        "validation_passed": result.validation_passed,
        "awk_fallback_used": result.awk_fallback_used,
        "highlights_detected": result.highlights_detected,
        "extracted_fields": result.extracted_fields,
        "quality_flags": result.quality_flags,
        "recommendations": result.recommendations,
        "stages_completed": [stage.value for stage in result.stages_completed],
        "errors": result.errors,
        "warnings": result.warnings,
    }

    output_file = Path(output_path)
    with output_file.open("w") as f:
        json.dump(result_dict, f, indent=2, default=str)


if __name__ == "__main__":
    app()
