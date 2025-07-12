"""Unified CLI - Main Command Interface

Comprehensive command-line interface for the unified vision processor using
the Llama 7-step pipeline with model selection and production features.
"""

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from ..config.unified_config import ModelType, UnifiedConfig
from ..evaluation import ComparisonConfiguration, ModelComparator
from ..extraction.hybrid_extraction_manager import UnifiedExtractionManager

# Initialize CLI app and console
app = typer.Typer(
    name="unified-vision-processor",
    help="Unified Vision Document Processing - Australian Tax Document Specialist",
    rich_markup_mode="rich",
)
console = Console()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.command()
def process(
    image_path: str = typer.Argument(..., help="Path to document image"),
    model: str = typer.Option(
        "internvl3",
        "--model",
        "-m",
        help="Vision model to use: internvl3 or llama32_vision",
    ),
    document_type: str | None = typer.Option(
        None,
        "--type",
        "-t",
        help="Document type (auto-detect if not specified)",
    ),
    output_path: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for results (JSON format)",
    ),
    format: str = typer.Option(
        "text",
        "--format",
        "-f",
        help="Output format: text or json",
    ),
    confidence_threshold: float = typer.Option(
        0.8,
        "--confidence-threshold",
        help="Minimum confidence threshold for results",
    ),
    enable_highlights: bool = typer.Option(
        False,
        "--enable-highlights/--no-highlights",
        help="Enable highlight detection",
    ),
    enable_awk_fallback: bool = typer.Option(
        False,
        "--enable-awk-fallback/--no-awk-fallback",
        help="Enable AWK fallback extraction",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
) -> None:
    """Process a single document using the unified Llama pipeline.

    Supports both InternVL3 and Llama-3.2-Vision models with identical
    7-step processing pipeline for fair comparison.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    console.print(f"[bold blue]🔬 Processing document with {model}[/bold blue]")
    console.print(f"Document: [green]{image_path}[/green]")

    try:
        # Validate model selection
        try:
            model_type = ModelType(model.lower())
        except ValueError:
            console.print(f"[red]❌ Invalid model: {model}[/red]")
            console.print(
                f"Available models: {', '.join([m.value for m in ModelType])}",
            )
            raise typer.Exit(1) from None

        # Validate image path
        image_file = Path(image_path)
        if not image_file.exists():
            console.print(f"[red]❌ Image file not found: {image_path}[/red]")
            raise typer.Exit(1) from None

        # Create configuration
        config = UnifiedConfig.from_env()
        config.model_type = model_type
        config.confidence_threshold = confidence_threshold
        config.highlight_detection = enable_highlights
        config.awk_fallback = enable_awk_fallback

        # Process document using unified pipeline
        with console.status(f"[bold green]Processing with {model_type.value}..."):
            with UnifiedExtractionManager(config) as extraction_manager:
                result = extraction_manager.process_document(image_file, document_type)

        # Display results based on format
        if format.lower() == "json":
            _display_json_result(result)
        else:
            _display_processing_result(result)

        # Save output if requested
        if output_path:
            _save_processing_result(result, output_path, format)
            console.print(f"[green]✅ Results saved to: {output_path}[/green]")

    except Exception as e:
        console.print(f"[red]❌ Processing failed: {e}[/red]")
        if verbose:
            import traceback

            console.print(traceback.format_exc())
        raise typer.Exit(1) from None


@app.command()
def batch(
    dataset_path: str = typer.Argument(..., help="Path to dataset directory"),
    model: str = typer.Option("internvl3", "--model", "-m", help="Vision model to use"),
    output_dir: str = typer.Option(
        "./output",
        "--output",
        "-o",
        help="Output directory for results",
    ),
    max_documents: int | None = typer.Option(
        None,
        "--max",
        "-n",
        help="Maximum number of documents to process",
    ),
    _workers: int = typer.Option(
        1,
        "--workers",
        "-w",
        help="Number of parallel workers",
    ),
    generate_report: bool = typer.Option(
        True,
        "--report/--no-report",
        help="Generate processing report",
    ),
) -> None:
    """Process multiple documents in batch using the unified pipeline.

    Provides comprehensive statistics and production readiness assessment
    using the Llama 5-level quality system.
    """
    console.print(f"[bold blue]📦 Batch processing with {model}[/bold blue]")
    console.print(f"Dataset: [green]{dataset_path}[/green]")

    try:
        # Validate inputs
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            console.print(f"[red]❌ Dataset directory not found: {dataset_path}[/red]")
            raise typer.Exit(1) from None

        output_directory = Path(output_dir)
        output_directory.mkdir(parents=True, exist_ok=True)

        # Find all images
        image_files = list(dataset_dir.glob("*.jpg")) + list(dataset_dir.glob("*.png"))
        if max_documents:
            image_files = image_files[:max_documents]

        console.print(f"Found [yellow]{len(image_files)}[/yellow] images to process")

        # Create configuration
        config = UnifiedConfig.from_env()
        config.model_type = ModelType(model.lower())

        # Process batch with progress tracking
        results = []

        with Progress() as progress:
            task = progress.add_task(
                f"[green]Processing with {model}...",
                total=len(image_files),
            )

            with UnifiedExtractionManager(config) as extraction_manager:
                for i, image_file in enumerate(image_files):
                    try:
                        result = extraction_manager.process_document(image_file)
                        results.append(
                            {
                                "image_file": image_file.name,
                                "success": True,
                                "result": result,
                            },
                        )

                        # Extract key fields for progress display (check both standard and AWK field names)
                        key_field_mappings = [
                            (["total_amount", "total_value"], "total"),
                            (["supplier_name", "supplier_value", "business_name"], "supplier"),
                            (["date", "date_value"], "date"),
                            (["gst_amount", "gst_value"], "gst"),
                        ]
                        extracted_info = []

                        for field_options, display_name in key_field_mappings:
                            value = None
                            for field in field_options:
                                value = result.extracted_fields.get(field)
                                if value:
                                    break

                            if value:
                                # Truncate long values
                                display_value = (
                                    str(value)[:20] + "..." if len(str(value)) > 20 else str(value)
                                )
                                extracted_info.append(f"{display_name}:{display_value}")

                        fields_summary = " | ".join(extracted_info) if extracted_info else "No key fields"

                        progress.update(
                            task,
                            advance=1,
                            description=f"[green]Processed {i + 1}/{len(image_files)} - {result.quality_grade.value} | {fields_summary}",
                        )

                    except Exception as e:
                        results.append(
                            {
                                "image_file": image_file.name,
                                "success": False,
                                "error": str(e),
                            },
                        )

                        progress.update(task, advance=1)
                        logger.warning(f"Failed to process {image_file.name}: {e}")

        # Display batch statistics
        _display_batch_statistics(results, model)

        # Save batch results
        batch_output_file = output_directory / f"batch_results_{model}.json"
        _save_batch_results(results, batch_output_file)

        # Generate report if requested
        if generate_report:
            report_file = output_directory / f"batch_report_{model}.html"
            _generate_batch_report(results, model, report_file)
            console.print(f"[green]📊 Report generated: {report_file}[/green]")

        console.print("[green]✅ Batch processing completed[/green]")

    except Exception as e:
        console.print(f"[red]❌ Batch processing failed: {e}[/red]")
        raise typer.Exit(1) from None


@app.command()
def compare(
    dataset_path: str = typer.Argument(..., help="Path to dataset directory"),
    ground_truth_path: str = typer.Argument(..., help="Path to ground truth directory"),
    models: str = typer.Option(
        "internvl3,llama32_vision",
        "--models",
        "-m",
        help="Comma-separated list of models to compare",
    ),
    output_dir: str = typer.Option(
        "./comparison_output",
        "--output",
        "-o",
        help="Output directory for comparison results",
    ),
    max_documents: int | None = typer.Option(
        None,
        "--max",
        "-n",
        help="Maximum number of documents to evaluate",
    ),
    report_format: str = typer.Option(
        "html",
        "--format",
        "-f",
        help="Report format: html, json, or text",
    ),
) -> None:
    """Compare multiple models using identical Llama pipeline for fairness.

    Ensures unbiased comparison by using the same 7-step processing pipeline
    for all models, eliminating architectural differences.
    """
    model_list = [m.strip() for m in models.split(",")]

    console.print("[bold blue]⚖️ Fair Model Comparison[/bold blue]")
    console.print(f"Models: [yellow]{', '.join(model_list)}[/yellow]")
    console.print(f"Dataset: [green]{dataset_path}[/green]")

    try:
        # Validate inputs
        dataset_dir = Path(dataset_path)
        gt_dir = Path(ground_truth_path)

        if not dataset_dir.exists():
            console.print(f"[red]❌ Dataset directory not found: {dataset_path}[/red]")
            raise typer.Exit(1) from None

        if not gt_dir.exists():
            console.print(
                f"[red]❌ Ground truth directory not found: {ground_truth_path}[/red]",
            )
            raise typer.Exit(1) from None

        output_directory = Path(output_dir)
        output_directory.mkdir(parents=True, exist_ok=True)

        # Create comparison configuration
        config = UnifiedConfig.from_env()
        comparator = ModelComparator(config)

        comparison_config = ComparisonConfiguration(
            models_to_compare=model_list,
            dataset_path=dataset_dir,
            ground_truth_path=gt_dir,
            max_documents=max_documents,
            identical_pipeline=True,  # Ensure fairness
            generate_reports=True,
            report_formats=[report_format],
            output_directory=output_directory,
        )

        # Perform comparison
        with console.status("[bold green]Performing fair model comparison..."):
            comparison_result = comparator.compare_models(comparison_config)

        # Display comparison results
        _display_comparison_results(comparison_result)

        console.print("[green]✅ Model comparison completed[/green]")
        console.print(f"Results saved to: [blue]{output_directory}[/blue]")

    except Exception as e:
        console.print(f"[red]❌ Model comparison failed: {e}[/red]")
        raise typer.Exit(1) from None


@app.command()
def evaluate(
    dataset_path: str = typer.Argument(..., help="Path to SROIE dataset"),
    ground_truth_path: str = typer.Argument(..., help="Path to SROIE ground truth"),
    model: str = typer.Option("internvl3", "--model", "-m", help="Model to evaluate"),
    output_dir: str = typer.Option(
        "./evaluation_output",
        "--output",
        "-o",
        help="Output directory for evaluation results",
    ),
    enhanced_fields: bool = typer.Option(
        True,
        "--enhanced/--standard",
        help="Use enhanced Australian tax fields",
    ),
) -> None:
    """Evaluate model on SROIE dataset with Australian tax enhancements.

    Provides specialized evaluation for receipt processing with ATO compliance
    and Australian business context validation.
    """
    console.print("[bold blue]📊 SROIE Dataset Evaluation[/bold blue]")
    console.print(f"Model: [yellow]{model}[/yellow]")
    console.print(f"Enhanced fields: [green]{enhanced_fields}[/green]")

    try:
        # Create configuration and evaluator
        config = UnifiedConfig.from_env()
        config.model_type = ModelType(model.lower())

        from ..evaluation import SROIEEvaluator

        evaluator = SROIEEvaluator(config)

        output_directory = Path(output_dir)
        output_directory.mkdir(parents=True, exist_ok=True)

        # Perform SROIE evaluation
        with console.status(f"[bold green]Evaluating {model} on SROIE dataset..."):
            evaluation_result = evaluator.evaluate_sroie_dataset(
                dataset_path,
                ground_truth_path,
                model,
                enhanced_fields,
            )

        # Display evaluation results
        _display_sroie_results(evaluation_result)

        # Save results
        results_file = output_directory / f"sroie_evaluation_{model}.json"
        import json

        with results_file.open("w") as f:
            json.dump(evaluation_result, f, indent=2, default=str)

        console.print("[green]✅ SROIE evaluation completed[/green]")
        console.print(f"Results saved to: [blue]{results_file}[/blue]")

    except Exception as e:
        console.print(f"[red]❌ SROIE evaluation failed: {e}[/red]")
        raise typer.Exit(1) from None


def _display_processing_result(result) -> None:
    """Display single document processing result."""
    # Create results table
    table = Table(title="🔍 Document Processing Results")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Status", justify="center")

    # Add basic info
    table.add_row("Model", result.model_type, "✅")
    table.add_row("Document Type", result.document_type, "✅")
    table.add_row("Processing Time", f"{result.processing_time:.2f}s", "⏱️")

    # Add confidence metrics
    table.add_row("Confidence Score", f"{result.confidence_score:.3f}", "📊")
    table.add_row("Quality Grade", result.quality_grade.value.title(), "🎯")
    table.add_row(
        "Production Ready",
        "✅ Yes" if result.production_ready else "❌ No",
        "🏭",
    )
    table.add_row("ATO Compliance", f"{result.ato_compliance_score:.3f}", "🇦🇺")

    # Add pipeline info
    table.add_row(
        "AWK Fallback Used",
        "Yes" if result.awk_fallback_used else "No",
        "🔄",
    )
    table.add_row("Highlights Detected", str(result.highlights_detected), "🔍")

    console.print(table)

    # Display extracted fields
    if result.extracted_fields:
        field_table = Table(title="📝 Extracted Fields")
        field_table.add_column("Field", style="cyan")
        field_table.add_column("Value", style="yellow")

        for field, value in result.extracted_fields.items():
            if not field.startswith("_") and value:
                field_table.add_row(field.replace("_", " ").title(), str(value))

        console.print(field_table)


def _display_batch_statistics(results: list[dict], model: str) -> None:
    """Display batch processing statistics."""
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    # Basic statistics
    stats_table = Table(title=f"📦 Batch Processing Statistics - {model}")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")

    stats_table.add_row("Total Documents", str(len(results)))
    stats_table.add_row("Successful", str(len(successful)))
    stats_table.add_row("Failed", str(len(failed)))
    stats_table.add_row("Success Rate", f"{len(successful) / len(results):.1%}")

    if successful:
        # Quality distribution
        quality_counts = {}
        production_ready = 0
        total_time = 0

        for result_entry in successful:
            result = result_entry["result"]
            grade = result.quality_grade.value
            quality_counts[grade] = quality_counts.get(grade, 0) + 1

            if result.production_ready:
                production_ready += 1

            total_time += result.processing_time

        stats_table.add_row(
            "Production Ready",
            f"{production_ready}/{len(successful)} ({production_ready / len(successful):.1%})",
        )
        stats_table.add_row(
            "Avg Processing Time",
            f"{total_time / len(successful):.2f}s",
        )

        # Quality distribution
        quality_str = ", ".join(f"{k}: {v}" for k, v in quality_counts.items())
        stats_table.add_row("Quality Distribution", quality_str)

    console.print(stats_table)

    # Show sample extracted fields from the first few successful results
    if successful and len(successful) >= 3:
        sample_table = Table(title="📋 Sample Extracted Fields (First 3 Documents)")
        sample_table.add_column("Document", style="cyan")
        sample_table.add_column("Total Amount", style="green")
        sample_table.add_column("Supplier", style="yellow")
        sample_table.add_column("Date", style="blue")
        sample_table.add_column("Fields Count", justify="center")

        for _i, result_entry in enumerate(successful[:3]):
            result = result_entry["result"]
            fields = result.extracted_fields

            # Extract key fields
            total_amount = fields.get("total_amount", "Not found")
            supplier = fields.get("supplier_name", "Not found")
            date = fields.get("date", "Not found")
            field_count = len([v for v in fields.values() if v and str(v).strip()])

            # Truncate long values for display
            supplier_display = str(supplier)[:25] + "..." if len(str(supplier)) > 25 else str(supplier)

            sample_table.add_row(
                result_entry["image_file"], str(total_amount), supplier_display, str(date), str(field_count)
            )

        console.print(sample_table)


def _display_comparison_results(comparison_result) -> None:
    """Display model comparison results."""
    rankings = comparison_result.performance_rankings

    # Performance rankings table
    ranking_table = Table(title="🏆 Model Performance Rankings")
    ranking_table.add_column("Rank", justify="center", style="bold")
    ranking_table.add_column("Model", style="cyan")
    ranking_table.add_column("F1 Score", justify="center")
    ranking_table.add_column("Production Ready", justify="center")
    ranking_table.add_column("Processing Speed", justify="center")
    ranking_table.add_column("Overall Score", justify="center")

    for ranking in rankings:
        ranking_table.add_row(
            str(ranking["overall_rank"]),
            ranking["model_name"],
            f"{ranking['metrics']['f1_score']:.3f}",
            f"{ranking['metrics']['production_ready_rate']:.1%}",
            f"{ranking['metrics']['processing_speed']:.2f}",
            f"{ranking['overall_score']:.1f}",
        )

    console.print(ranking_table)

    # Deployment recommendation
    deployment = comparison_result.deployment_recommendations.get(
        "production_deployment",
        {},
    )

    if deployment.get("status") == "ready":
        console.print(
            f"[green]✅ Recommended for production: {deployment['model']}[/green]",
        )
    elif deployment.get("status") == "conditional":
        console.print(
            f"[yellow]⚠️ Conditional recommendation: {deployment['model']}[/yellow]",
        )
    else:
        console.print("[red]❌ No model ready for production deployment[/red]")


def _display_sroie_results(evaluation_result: dict) -> None:
    """Display SROIE evaluation results."""
    overall = evaluation_result["overall_metrics"]

    # Overall metrics table
    overall_table = Table(title="📊 SROIE Evaluation Results")
    overall_table.add_column("Metric", style="cyan")
    overall_table.add_column("Value", style="green")

    overall_table.add_row("Overall F1 Score", f"{overall['average_f1']:.3f}")
    overall_table.add_row("Success Rate", f"{overall['success_rate']:.1%}")
    overall_table.add_row("Total Documents", str(overall["total_documents"]))
    overall_table.add_row(
        "Successful Extractions",
        str(overall["successful_documents"]),
    )

    console.print(overall_table)

    # Field-specific metrics
    sroie_metrics = evaluation_result["sroie_metrics"]

    field_table = Table(title="📝 Field-Specific Performance")
    field_table.add_column("Field", style="cyan")
    field_table.add_column("F1 Score", justify="center")
    field_table.add_column("Precision", justify="center")
    field_table.add_column("Recall", justify="center")
    field_table.add_column("Samples", justify="center")

    for field, metrics in sroie_metrics.items():
        field_table.add_row(
            field.title(),
            f"{metrics['f1']:.3f}",
            f"{metrics['precision']:.3f}",
            f"{metrics['recall']:.3f}",
            str(metrics["sample_count"]),
        )

    console.print(field_table)


def _display_json_result(result) -> None:
    """Display processing result in JSON format."""
    import json

    result_dict = {
        "model_type": result.model_type,
        "document_type": result.document_type,
        "processing_time": result.processing_time,
        "confidence_score": result.confidence_score,
        "quality_grade": result.quality_grade.value,
        "production_ready": result.production_ready,
        "ato_compliance_score": result.ato_compliance_score,
        "extracted_fields": result.extracted_fields,
        "awk_fallback_used": result.awk_fallback_used,
        "highlights_detected": result.highlights_detected,
        "quality_flags": result.quality_flags,
        "recommendations": result.recommendations,
    }

    console.print(json.dumps(result_dict, indent=2, default=str))


def _save_processing_result(result, output_path: str, format: str = "json") -> None:
    """Save processing result to file."""
    import json

    result_dict = {
        "model_type": result.model_type,
        "document_type": result.document_type,
        "processing_time": result.processing_time,
        "confidence_score": result.confidence_score,
        "quality_grade": result.quality_grade.value,
        "production_ready": result.production_ready,
        "ato_compliance_score": result.ato_compliance_score,
        "extracted_fields": result.extracted_fields,
        "awk_fallback_used": result.awk_fallback_used,
        "highlights_detected": result.highlights_detected,
        "quality_flags": result.quality_flags,
        "recommendations": result.recommendations,
    }

    if format.lower() == "json":
        with Path(output_path).open("w") as f:
            json.dump(result_dict, f, indent=2, default=str)
    else:
        # Text format
        with Path(output_path).open("w") as f:
            f.write(f"Model: {result.model_type}\n")
            f.write(f"Document Type: {result.document_type}\n")
            f.write(f"Processing Time: {result.processing_time:.2f}s\n")
            f.write(f"Confidence Score: {result.confidence_score:.3f}\n")
            f.write(f"Quality Grade: {result.quality_grade.value}\n")
            f.write(f"Production Ready: {result.production_ready}\n")
            f.write("\nExtracted Fields:\n")
            for field, value in result.extracted_fields.items():
                if not field.startswith("_") and value:
                    f.write(f"  {field}: {value}\n")


def _save_batch_results(results: list[dict], output_file: Path) -> None:
    """Save batch results to JSON file."""
    import json

    # Convert results to JSON-serializable format
    serializable_results = []
    for result_entry in results:
        if result_entry["success"]:
            result = result_entry["result"]
            serializable_entry = {
                "image_file": result_entry["image_file"],
                "success": True,
                "model_type": result.model_type,
                "document_type": result.document_type,
                "processing_time": result.processing_time,
                "confidence_score": result.confidence_score,
                "quality_grade": result.quality_grade.value,
                "production_ready": result.production_ready,
                "extracted_fields": result.extracted_fields,
            }
        else:
            serializable_entry = {
                "image_file": result_entry["image_file"],
                "success": False,
                "error": result_entry["error"],
            }

        serializable_results.append(serializable_entry)

    with output_file.open("w") as f:
        json.dump(serializable_results, f, indent=2, default=str)


def _generate_batch_report(results: list[dict], model: str, report_file: Path) -> None:
    """Generate HTML batch processing report."""
    successful = [r for r in results if r["success"]]

    if not successful:
        return

    # Calculate statistics
    quality_counts = {}
    production_ready = 0
    total_time = sum(r["result"].processing_time for r in successful)

    for result_entry in successful:
        result = result_entry["result"]
        grade = result.quality_grade.value
        quality_counts[grade] = quality_counts.get(grade, 0) + 1

        if result.production_ready:
            production_ready += 1

    # Generate simple HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Batch Processing Report - {model}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .summary {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .metric {{ margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>Batch Processing Report - {model}</h1>
        <div class="summary">
            <h2>Summary Statistics</h2>
            <div class="metric">Total Documents: {len(results)}</div>
            <div class="metric">Successful: {len(successful)}</div>
            <div class="metric">Success Rate: {len(successful) / len(results):.1%}</div>
            <div class="metric">Production Ready: {production_ready}/{len(successful)} ({production_ready / len(successful):.1%})</div>
            <div class="metric">Average Processing Time: {total_time / len(successful):.2f}s</div>
            <div class="metric">Quality Distribution: {", ".join(f"{k}: {v}" for k, v in quality_counts.items())}</div>
        </div>
    </body>
    </html>
    """

    with report_file.open("w") as f:
        f.write(html_content)


if __name__ == "__main__":
    app()
