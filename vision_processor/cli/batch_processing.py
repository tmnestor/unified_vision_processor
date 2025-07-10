"""Batch Processing CLI

Scalable batch processing interface with comprehensive statistics,
production monitoring, and parallel processing capabilities.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from ..config.unified_config import ModelType, UnifiedConfig
from ..extraction.hybrid_extraction_manager import UnifiedExtractionManager

# Initialize app and console
app = typer.Typer(
    name="batch-processing",
    help="Scalable batch processing with production monitoring",
    rich_markup_mode="rich",
)
console = Console()

logger = logging.getLogger(__name__)


@app.command()
def process(
    dataset_path: str = typer.Argument(..., help="Path to dataset directory"),
    model: str = typer.Option(
        "internvl3",
        "--model",
        "-m",
        help="Vision model: internvl3 or llama32_vision",
    ),
    output_dir: str = typer.Option(
        "./batch_output",
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
    workers: int = typer.Option(
        4,
        "--workers",
        "-w",
        help="Number of parallel workers (1-16)",
    ),
    confidence_threshold: float = typer.Option(
        0.7,
        "--threshold",
        "-t",
        help="Confidence threshold for production readiness",
    ),
    generate_report: bool = typer.Option(
        True,
        "--report/--no-report",
        help="Generate comprehensive HTML report",
    ),
    monitor_quality: bool = typer.Option(
        True,
        "--monitor/--no-monitor",
        help="Enable real-time quality monitoring",
    ),
    fail_fast: bool = typer.Option(
        False,
        "--fail-fast/--continue-on-error",
        help="Stop processing on first error",
    ),
) -> None:
    """Process multiple documents with production monitoring.

    Provides comprehensive batch processing with parallel execution,
    real-time quality monitoring, and detailed statistics using the
    Llama 5-level production readiness assessment.
    """
    # Validate inputs
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        console.print(f"[red]❌ Dataset directory not found: {dataset_path}[/red]")
        raise typer.Exit(1) from None

    try:
        model_type = ModelType(model.lower())
    except ValueError:
        console.print(f"[red]❌ Invalid model: {model}[/red]")
        raise typer.Exit(1) from None

    # Validate workers
    workers = max(1, min(workers, 16))

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel.fit(
            f"[bold blue]📦 Batch Processing Setup[/bold blue]\n"
            f"Dataset: [green]{dataset_dir}[/green]\n"
            f"Model: [yellow]{model_type.value}[/yellow]\n"
            f"Workers: [cyan]{workers}[/cyan]\n"
            f"Output: [blue]{output_directory}[/blue]",
            title="Batch Configuration",
        ),
    )

    try:
        # Find all images
        image_files = _find_image_files(dataset_dir, max_documents)

        if not image_files:
            console.print("[red]❌ No image files found in dataset directory[/red]")
            raise typer.Exit(1) from None

        console.print(f"[green]📸 Found {len(image_files)} images to process[/green]")

        # Create configuration
        config = UnifiedConfig.from_env()
        config.model_type = model_type
        config.confidence_threshold = confidence_threshold

        # Process batch
        start_time = time.time()

        if workers == 1:
            results = _process_sequential(
                image_files,
                config,
                monitor_quality,
                fail_fast,
            )
        else:
            results = _process_parallel(
                image_files,
                config,
                workers,
                monitor_quality,
                fail_fast,
            )

        total_time = time.time() - start_time

        # Analyze results
        analysis = _analyze_batch_results(results, total_time, confidence_threshold)

        # Display comprehensive statistics
        _display_batch_analysis(analysis, model_type.value)

        # Save results
        results_file = (
            output_directory / f"batch_results_{model}_{int(time.time())}.json"
        )
        _save_batch_results(results, analysis, results_file)

        # Generate report if requested
        if generate_report:
            report_file = (
                output_directory / f"batch_report_{model}_{int(time.time())}.html"
            )
            _generate_comprehensive_report(
                results,
                analysis,
                model_type.value,
                report_file,
            )
            console.print(f"[green]📊 Comprehensive report: {report_file}[/green]")

        # Production readiness summary
        _display_production_summary(analysis)

        console.print(
            f"[green]✅ Batch processing completed in {total_time:.1f}s[/green]",
        )

        # Exit with appropriate code
        if analysis["production_readiness"]["overall_ready"]:
            console.print("[green]🚀 Batch ready for production deployment[/green]")
        else:
            console.print("[yellow]⚠️ Batch requires review before production[/yellow]")
            typer.Exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Processing interrupted by user[/yellow]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]❌ Batch processing failed: {e}[/red]")
        raise typer.Exit(1) from None


@app.command()
def monitor(
    dataset_path: str = typer.Argument(..., help="Path to dataset directory"),
    model: str = typer.Option("internvl3", "--model", "-m", help="Vision model to use"),
    sample_size: int = typer.Option(
        10,
        "--sample",
        "-s",
        help="Number of documents to sample for monitoring",
    ),
    refresh_interval: int = typer.Option(
        30,
        "--interval",
        "-i",
        help="Monitoring refresh interval in seconds",
    ),
) -> None:
    """Real-time production monitoring of document processing.

    Continuously monitors processing quality and performance metrics
    for production deployment validation.
    """
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        console.print(f"[red]❌ Dataset directory not found: {dataset_path}[/red]")
        raise typer.Exit(1) from None

    try:
        model_type = ModelType(model.lower())
    except ValueError:
        console.print(f"[red]❌ Invalid model: {model}[/red]")
        raise typer.Exit(1) from None

    console.print(
        Panel.fit(
            f"[bold blue]🔍 Production Monitoring[/bold blue]\n"
            f"Model: [yellow]{model_type.value}[/yellow]\n"
            f"Sample Size: [cyan]{sample_size}[/cyan]\n"
            f"Refresh: [green]{refresh_interval}s[/green]",
            title="Monitoring Configuration",
        ),
    )

    try:
        _run_production_monitoring(
            dataset_dir,
            model_type,
            sample_size,
            refresh_interval,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]📊 Monitoring stopped[/yellow]")


@app.command()
def analyze(
    results_file: str = typer.Argument(..., help="Path to batch results JSON file"),
    output_format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, json, detailed",
    ),
    export_csv: str | None = typer.Option(
        None,
        "--csv",
        help="Export statistics to CSV file",
    ),
) -> None:
    """Analyze existing batch processing results.

    Provides detailed analysis of batch processing results including
    quality distributions, performance metrics, and recommendations.
    """
    results_path = Path(results_file)
    if not results_path.exists():
        console.print(f"[red]❌ Results file not found: {results_file}[/red]")
        raise typer.Exit(1) from None

    try:
        with Path(results_path).open("r") as f:
            data = json.load(f)

        results = data.get("results", [])
        analysis = data.get("analysis", {})

        if not results:
            console.print("[red]❌ No results found in file[/red]")
            raise typer.Exit(1) from None

        # Display analysis based on format
        if output_format == "json":
            console.print(json.dumps(analysis, indent=2))
        elif output_format == "detailed":
            _display_detailed_analysis(results, analysis)
        else:  # table format
            _display_batch_analysis(analysis, "Analysis")

        # Export to CSV if requested
        if export_csv:
            _export_to_csv(results, export_csv)
            console.print(f"[green]📄 Statistics exported to: {export_csv}[/green]")

    except Exception as e:
        console.print(f"[red]❌ Analysis failed: {e}[/red]")
        raise typer.Exit(1) from None


def _find_image_files(dataset_dir: Path, max_documents: int | None) -> list[Path]:
    """Find all image files in dataset directory."""
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    image_files = []

    for ext in extensions:
        image_files.extend(dataset_dir.glob(ext))
        image_files.extend(dataset_dir.glob(ext.upper()))

    # Sort for consistent processing order
    image_files.sort()

    if max_documents:
        image_files = image_files[:max_documents]

    return image_files


def _process_sequential(
    image_files: list[Path],
    config: UnifiedConfig,
    monitor_quality: bool,
    fail_fast: bool,
) -> list[dict]:
    """Process documents sequentially with progress tracking."""
    results = []

    # Create progress display
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    quality_stats = {"excellent": 0, "good": 0, "fair": 0, "poor": 0, "very_poor": 0}

    with progress:
        task = progress.add_task("Processing documents...", total=len(image_files))

        with UnifiedExtractionManager(config) as extraction_manager:
            for i, image_file in enumerate(image_files):
                try:
                    result = extraction_manager.process_document(image_file)

                    # Update quality stats
                    quality_stats[result.quality_grade.value] += 1

                    results.append(
                        {
                            "image_file": image_file.name,
                            "success": True,
                            "result": result,
                            "processing_order": i,
                        },
                    )

                    # Update progress with quality info
                    if monitor_quality:
                        quality_summary = f"Excellent: {quality_stats['excellent']}, Good: {quality_stats['good']}, Fair: {quality_stats['fair']}"
                        progress.update(
                            task,
                            advance=1,
                            description=f"Processing... ({quality_summary})",
                        )
                    else:
                        progress.update(task, advance=1)

                except Exception as e:
                    results.append(
                        {
                            "image_file": image_file.name,
                            "success": False,
                            "error": str(e),
                            "processing_order": i,
                        },
                    )

                    if fail_fast:
                        logger.error(f"Failing fast due to error: {e}")
                        break

                    progress.update(task, advance=1)
                    logger.warning(f"Failed to process {image_file.name}: {e}")

    return results


def _process_parallel(
    image_files: list[Path],
    config: UnifiedConfig,
    workers: int,
    monitor_quality: bool,
    fail_fast: bool,
) -> list[dict]:
    """Process documents in parallel with progress tracking."""
    results = []

    # Create progress display
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    quality_stats = {"excellent": 0, "good": 0, "fair": 0, "poor": 0, "very_poor": 0}

    with progress:
        task = progress.add_task(
            f"Processing with {workers} workers...",
            total=len(image_files),
        )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_image = {}
            for i, image_file in enumerate(image_files):
                future = executor.submit(
                    _process_single_document,
                    image_file,
                    config,
                    i,
                )
                future_to_image[future] = image_file

            # Collect results as they complete
            for future in as_completed(future_to_image):
                image_file = future_to_image[future]

                try:
                    result_data = future.result()
                    results.append(result_data)

                    # Update quality stats for successful results
                    if result_data["success"]:
                        quality_grade = result_data["result"].quality_grade.value
                        quality_stats[quality_grade] += 1

                    # Update progress
                    if monitor_quality:
                        total_processed = sum(quality_stats.values())
                        if total_processed > 0:
                            excellent_rate = (
                                quality_stats["excellent"] / total_processed * 100
                            )
                            progress.update(
                                task,
                                advance=1,
                                description=f"Processing... (Excellent: {excellent_rate:.1f}%)",
                            )
                        else:
                            progress.update(task, advance=1)
                    else:
                        progress.update(task, advance=1)

                except Exception as e:
                    results.append(
                        {
                            "image_file": image_file.name,
                            "success": False,
                            "error": str(e),
                            "processing_order": -1,
                        },
                    )

                    if fail_fast:
                        # Cancel remaining futures
                        for remaining_future in future_to_image:
                            remaining_future.cancel()
                        break

                    progress.update(task, advance=1)
                    logger.warning(f"Failed to process {image_file.name}: {e}")

    # Sort results by processing order to maintain consistency
    results.sort(key=lambda x: x.get("processing_order", 999999))

    return results


def _process_single_document(
    image_file: Path,
    config: UnifiedConfig,
    order: int,
) -> dict:
    """Process a single document (for parallel execution)."""
    try:
        with UnifiedExtractionManager(config) as extraction_manager:
            result = extraction_manager.process_document(image_file)

            return {
                "image_file": image_file.name,
                "success": True,
                "result": result,
                "processing_order": order,
            }

    except Exception as e:
        return {
            "image_file": image_file.name,
            "success": False,
            "error": str(e),
            "processing_order": order,
        }


def _analyze_batch_results(
    results: list[dict],
    total_time: float,
    confidence_threshold: float,
) -> dict:
    """Analyze batch processing results comprehensively."""
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]

    analysis = {
        "summary": {
            "total_documents": len(results),
            "successful": len(successful_results),
            "failed": len(failed_results),
            "success_rate": len(successful_results) / len(results) if results else 0,
            "total_processing_time": total_time,
            "average_processing_time": total_time / len(results) if results else 0,
        },
        "quality_distribution": {},
        "confidence_analysis": {},
        "production_readiness": {},
        "performance_metrics": {},
        "ato_compliance": {},
        "error_analysis": {},
    }

    if successful_results:
        # Quality distribution
        quality_counts = {}
        confidence_scores = []
        processing_times = []
        ato_scores = []
        production_ready_count = 0
        awk_fallback_count = 0
        highlights_total = 0

        for result_entry in successful_results:
            result = result_entry["result"]

            # Quality distribution
            grade = result.quality_grade.value
            quality_counts[grade] = quality_counts.get(grade, 0) + 1

            # Metrics collection
            confidence_scores.append(result.confidence_score)
            processing_times.append(result.processing_time)
            ato_scores.append(result.ato_compliance_score)

            if result.production_ready:
                production_ready_count += 1

            if result.awk_fallback_used:
                awk_fallback_count += 1

            highlights_total += result.highlights_detected

        analysis["quality_distribution"] = quality_counts

        # Confidence analysis
        analysis["confidence_analysis"] = {
            "average_confidence": sum(confidence_scores) / len(confidence_scores),
            "min_confidence": min(confidence_scores),
            "max_confidence": max(confidence_scores),
            "above_threshold": sum(
                1 for c in confidence_scores if c >= confidence_threshold
            ),
            "below_threshold": sum(
                1 for c in confidence_scores if c < confidence_threshold
            ),
        }

        # Production readiness
        production_rate = production_ready_count / len(successful_results)
        analysis["production_readiness"] = {
            "ready_count": production_ready_count,
            "ready_rate": production_rate,
            "overall_ready": production_rate
            >= 0.8,  # 80% threshold for batch readiness
            "confidence_above_threshold": analysis["confidence_analysis"][
                "above_threshold"
            ],
        }

        # Performance metrics
        analysis["performance_metrics"] = {
            "average_processing_time": sum(processing_times) / len(processing_times),
            "min_processing_time": min(processing_times),
            "max_processing_time": max(processing_times),
            "documents_per_second": len(successful_results) / total_time,
            "awk_fallback_rate": awk_fallback_count / len(successful_results),
            "average_highlights": highlights_total / len(successful_results),
        }

        # ATO compliance
        analysis["ato_compliance"] = {
            "average_score": sum(ato_scores) / len(ato_scores),
            "min_score": min(ato_scores),
            "max_score": max(ato_scores),
            "high_compliance": sum(1 for s in ato_scores if s >= 0.8),
        }

    # Error analysis
    if failed_results:
        error_types = {}
        for result_entry in failed_results:
            error = result_entry.get("error", "Unknown error")
            error_type = error.split(":")[0] if ":" in error else error
            error_types[error_type] = error_types.get(error_type, 0) + 1

        analysis["error_analysis"] = {
            "error_types": error_types,
            "failed_files": [r["image_file"] for r in failed_results],
        }

    return analysis


def _display_batch_analysis(analysis: dict, model_name: str) -> None:
    """Display comprehensive batch analysis."""
    # Summary table
    summary_table = Table(title=f"📊 Batch Processing Summary - {model_name}")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    summary_table.add_column("Status", justify="center")

    summary = analysis["summary"]
    summary_table.add_row("Total Documents", str(summary["total_documents"]), "📄")
    summary_table.add_row(
        "Successful",
        f"{summary['successful']}/{summary['total_documents']}",
        "✅",
    )
    summary_table.add_row(
        "Success Rate",
        f"{summary['success_rate']:.1%}",
        "🟢"
        if summary["success_rate"] >= 0.95
        else "🟡"
        if summary["success_rate"] >= 0.8
        else "🔴",
    )
    summary_table.add_row("Total Time", f"{summary['total_processing_time']:.1f}s", "⏱️")
    summary_table.add_row(
        "Avg Time/Doc",
        f"{summary['average_processing_time']:.2f}s",
        "📈",
    )

    console.print(summary_table)

    # Quality distribution
    if analysis.get("quality_distribution"):
        quality_table = Table(title="🎯 Quality Distribution")
        quality_table.add_column("Quality Grade", style="cyan")
        quality_table.add_column("Count", justify="center")
        quality_table.add_column("Percentage", justify="center")

        total_successful = sum(analysis["quality_distribution"].values())

        for grade in ["excellent", "good", "fair", "poor", "very_poor"]:
            count = analysis["quality_distribution"].get(grade, 0)
            percentage = count / total_successful * 100 if total_successful > 0 else 0

            grade_emoji = {
                "excellent": "🟢",
                "good": "🟡",
                "fair": "🟠",
                "poor": "🔴",
                "very_poor": "⚫",
            }

            quality_table.add_row(
                f"{grade_emoji.get(grade, '❓')} {grade.title()}",
                str(count),
                f"{percentage:.1f}%",
            )

        console.print(quality_table)

    # Performance metrics
    if "performance_metrics" in analysis:
        perf = analysis["performance_metrics"]
        perf_table = Table(title="⚡ Performance Metrics")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="yellow")

        perf_table.add_row("Documents/Second", f"{perf['documents_per_second']:.2f}")
        perf_table.add_row("AWK Fallback Rate", f"{perf['awk_fallback_rate']:.1%}")
        perf_table.add_row("Avg Highlights/Doc", f"{perf['average_highlights']:.1f}")
        perf_table.add_row(
            "Processing Range",
            f"{perf['min_processing_time']:.2f}s - {perf['max_processing_time']:.2f}s",
        )

        console.print(perf_table)


def _display_production_summary(analysis: dict) -> None:
    """Display production readiness summary."""
    if "production_readiness" not in analysis:
        return

    prod = analysis["production_readiness"]

    if prod["overall_ready"]:
        status_color = "green"
        status_icon = "🚀"
        status_text = "READY FOR PRODUCTION"
    else:
        status_color = "yellow"
        status_icon = "⚠️"
        status_text = "REQUIRES REVIEW"

    summary_content = f"""[bold {status_color}]{status_icon} {status_text}[/bold {status_color}]

[bold]Production Ready Documents:[/bold] {prod["ready_count"]} ({prod["ready_rate"]:.1%})
[bold]High Confidence Documents:[/bold] {prod["confidence_above_threshold"]}
[bold]Batch Ready:[/bold] {"Yes" if prod["overall_ready"] else "No"}

[bold]Recommendations:[/bold]
"""

    if prod["overall_ready"]:
        summary_content += "• ✅ Batch approved for production deployment\n"
        summary_content += "• 📊 Monitor initial production performance\n"
        summary_content += "• 🔍 Review any failed documents separately"
    else:
        summary_content += "• ⚠️ Review documents below confidence threshold\n"
        summary_content += "• 🔧 Consider model fine-tuning if needed\n"
        summary_content += "• 📈 Investigate common failure patterns"

    console.print(Panel(summary_content, title="🏭 Production Readiness Assessment"))


def _run_production_monitoring(
    dataset_dir: Path,
    model_type: ModelType,
    sample_size: int,
    refresh_interval: int,
) -> None:
    """Run real-time production monitoring."""
    config = UnifiedConfig.from_env()
    config.model_type = model_type

    # Create monitoring table
    monitoring_table = Table(title="🔍 Real-time Production Monitoring")
    monitoring_table.add_column("Metric", style="cyan")
    monitoring_table.add_column("Current", style="green")
    monitoring_table.add_column("Target", style="yellow")
    monitoring_table.add_column("Status", justify="center")

    with Live(monitoring_table, console=console, refresh_per_second=0.5):
        iteration = 0
        while True:
            try:
                # Sample documents
                image_files = _find_image_files(dataset_dir, sample_size)

                if not image_files:
                    console.print("[red]No images found for monitoring[/red]")
                    break

                # Process sample
                start_time = time.time()
                results = _process_sequential(image_files, config, False, False)
                processing_time = time.time() - start_time

                # Analyze sample
                analysis = _analyze_batch_results(
                    results,
                    processing_time,
                    config.confidence_threshold,
                )

                # Update monitoring table
                monitoring_table.rows.clear()

                summary = analysis["summary"]
                prod = analysis.get("production_readiness", {})

                # Success rate
                success_rate = summary["success_rate"]
                monitoring_table.add_row(
                    "Success Rate",
                    f"{success_rate:.1%}",
                    "≥95%",
                    "🟢"
                    if success_rate >= 0.95
                    else "🟡"
                    if success_rate >= 0.8
                    else "🔴",
                )

                # Production ready rate
                prod_rate = prod.get("ready_rate", 0)
                monitoring_table.add_row(
                    "Production Ready",
                    f"{prod_rate:.1%}",
                    "≥80%",
                    "🟢" if prod_rate >= 0.8 else "🟡" if prod_rate >= 0.6 else "🔴",
                )

                # Processing speed
                docs_per_sec = analysis.get("performance_metrics", {}).get(
                    "documents_per_second",
                    0,
                )
                monitoring_table.add_row(
                    "Processing Speed",
                    f"{docs_per_sec:.2f} docs/s",
                    "≥0.5 docs/s",
                    "🟢"
                    if docs_per_sec >= 0.5
                    else "🟡"
                    if docs_per_sec >= 0.2
                    else "🔴",
                )

                # Confidence
                avg_confidence = analysis.get("confidence_analysis", {}).get(
                    "average_confidence",
                    0,
                )
                monitoring_table.add_row(
                    "Avg Confidence",
                    f"{avg_confidence:.3f}",
                    "≥0.7",
                    "🟢"
                    if avg_confidence >= 0.7
                    else "🟡"
                    if avg_confidence >= 0.5
                    else "🔴",
                )

                # Update iteration
                iteration += 1
                monitoring_table.title = (
                    f"🔍 Real-time Monitoring (Iteration {iteration})"
                )

                # Sleep until next refresh
                time.sleep(refresh_interval)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(refresh_interval)


def _save_batch_results(results: list[dict], analysis: dict, output_file: Path) -> None:
    """Save batch results and analysis to JSON file."""
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
                "ato_compliance_score": result.ato_compliance_score,
                "awk_fallback_used": result.awk_fallback_used,
                "highlights_detected": result.highlights_detected,
                "extracted_fields": result.extracted_fields,
                "quality_flags": result.quality_flags,
                "recommendations": result.recommendations,
            }
        else:
            serializable_entry = {
                "image_file": result_entry["image_file"],
                "success": False,
                "error": result_entry["error"],
            }

        serializable_results.append(serializable_entry)

    # Save comprehensive data
    output_data = {
        "results": serializable_results,
        "analysis": analysis,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with output_file.open("w") as f:
        json.dump(output_data, f, indent=2, default=str)


def _generate_comprehensive_report(
    results: list[dict],
    analysis: dict,
    model_name: str,
    report_file: Path,
) -> None:
    """Generate comprehensive HTML report."""
    from datetime import datetime

    # Calculate additional statistics
    [r for r in results if r["success"]]

    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Batch Processing Report - {model_name}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                   line-height: 1.6; margin: 0; padding: 20px; background-color: #f8f9fa; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; 
                        padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                           gap: 20px; margin: 30px 0; }}
            .metric-card {{ border: 1px solid #e1e5e9; border-radius: 8px; padding: 20px; background: #f8f9fa; }}
            .metric-value {{ font-size: 2em; font-weight: bold; color: #28a745; }}
            .metric-label {{ color: #6c757d; margin-top: 5px; }}
            .status-ready {{ color: #28a745; font-weight: bold; }}
            .status-warning {{ color: #ffc107; font-weight: bold; }}
            .status-error {{ color: #dc3545; font-weight: bold; }}
            .quality-bar {{ height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; }}
            .quality-segment {{ height: 100%; float: left; }}
            .excellent {{ background: #28a745; }}
            .good {{ background: #17a2b8; }}
            .fair {{ background: #ffc107; }}
            .poor {{ background: #fd7e14; }}
            .very-poor {{ background: #dc3545; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
            th {{ background-color: #f8f9fa; font-weight: 600; }}
            .timestamp {{ color: #6c757d; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📦 Batch Processing Report - {model_name}</h1>
            <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>📊 Executive Summary</h2>
            <div class="summary-grid">
                <div class="metric-card">
                    <div class="metric-value">{analysis["summary"]["total_documents"]}</div>
                    <div class="metric-label">Total Documents</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{analysis["summary"]["success_rate"]:.1%}</div>
                    <div class="metric-label">Success Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{analysis.get("production_readiness", {}).get("ready_rate", 0):.1%}</div>
                    <div class="metric-label">Production Ready</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{analysis.get("performance_metrics", {}).get("documents_per_second", 0):.2f}</div>
                    <div class="metric-label">Docs/Second</div>
                </div>
            </div>
            
            <h2>🎯 Quality Distribution</h2>
            <div class="quality-bar">
    """

    # Add quality distribution bars
    if "quality_distribution" in analysis:
        total_successful = sum(analysis["quality_distribution"].values())
        if total_successful > 0:
            for grade in ["excellent", "good", "fair", "poor", "very_poor"]:
                count = analysis["quality_distribution"].get(grade, 0)
                percentage = count / total_successful * 100
                html_content += f'<div class="quality-segment {grade}" style="width: {percentage}%;" title="{grade.title()}: {count} ({percentage:.1f}%)"></div>'

    html_content += f"""
            </div>
            
            <h2>📈 Performance Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Average Processing Time</td><td>{analysis.get("performance_metrics", {}).get("average_processing_time", 0):.2f}s</td></tr>
                <tr><td>AWK Fallback Rate</td><td>{analysis.get("performance_metrics", {}).get("awk_fallback_rate", 0):.1%}</td></tr>
                <tr><td>Average Confidence</td><td>{analysis.get("confidence_analysis", {}).get("average_confidence", 0):.3f}</td></tr>
                <tr><td>ATO Compliance</td><td>{analysis.get("ato_compliance", {}).get("average_score", 0):.3f}</td></tr>
            </table>
            
            <h2>🏭 Production Readiness</h2>
    """

    # Production readiness status
    prod_ready = analysis.get("production_readiness", {}).get("overall_ready", False)
    status_class = "status-ready" if prod_ready else "status-warning"
    status_text = "✅ Ready for Production" if prod_ready else "⚠️ Requires Review"

    html_content += f'<p class="{status_class}">{status_text}</p>'

    html_content += """
        </div>
    </body>
    </html>
    """

    with report_file.open("w", encoding="utf-8") as f:
        f.write(html_content)


def _display_detailed_analysis(results: list[dict], analysis: dict) -> None:
    """Display detailed analysis with individual document results."""
    _display_batch_analysis(analysis, "Detailed Analysis")

    # Document-level results
    successful_results = [r for r in results if r["success"]]

    if successful_results:
        doc_table = Table(title="📄 Individual Document Results (Top 10)")
        doc_table.add_column("Document", style="cyan")
        doc_table.add_column("Type", style="yellow")
        doc_table.add_column("Confidence", justify="center")
        doc_table.add_column("Quality", justify="center")
        doc_table.add_column("Production", justify="center")

        # Show top 10 by confidence
        sorted_results = sorted(
            successful_results,
            key=lambda x: x["result"].confidence_score,
            reverse=True,
        )

        for result_entry in sorted_results[:10]:
            result = result_entry["result"]
            doc_table.add_row(
                result_entry["image_file"],
                result.document_type.replace("_", " ").title(),
                f"{result.confidence_score:.3f}",
                result.quality_grade.value.title(),
                "✅" if result.production_ready else "❌",
            )

        console.print(doc_table)


def _export_to_csv(results: list[dict], csv_file: str) -> None:
    """Export results to CSV file."""
    import csv

    successful_results = [r for r in results if r["success"]]

    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(
            [
                "Image File",
                "Document Type",
                "Processing Time",
                "Confidence Score",
                "Quality Grade",
                "Production Ready",
                "ATO Compliance",
                "AWK Fallback",
                "Highlights Detected",
            ],
        )

        # Write data
        for result_entry in successful_results:
            result = result_entry["result"]
            writer.writerow(
                [
                    result_entry["image_file"],
                    result.document_type,
                    result.processing_time,
                    result.confidence_score,
                    result.quality_grade.value,
                    result.production_ready,
                    result.ato_compliance_score,
                    result.awk_fallback_used,
                    result.highlights_detected,
                ],
            )


if __name__ == "__main__":
    app()
