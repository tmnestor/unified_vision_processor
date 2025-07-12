"""Simplified CLI with deferred imports for testing."""

import os
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    name="unified-vision-processor",
    help="Unified Vision Document Processing - Australian Tax Document Specialist (Test Mode)",
    rich_markup_mode="rich",
)
console = Console()


@app.command()
def process(
    image_path: str = typer.Argument(..., help="Path to document image"),
    model: str = typer.Option(
        "internvl3",
        "--model",
        "-m",
        help="Vision model to use: internvl3 or llama32_vision",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
) -> None:
    """Process a single document using the unified pipeline."""
    if verbose:
        console.print(f"[bold blue]🔬 Processing document with {model}[/bold blue]")
        console.print(f"Document: [green]{image_path}[/green]")

    # Validate image path
    image_file = Path(image_path)
    if not image_file.exists():
        console.print(f"[red]❌ Image file not found: {image_path}[/red]")
        raise typer.Exit(1)

    try:
        # Import dependencies only when needed
        console.print("[yellow]Loading dependencies...[/yellow]")

        # Check environment variable
        if not os.environ.get("KMP_DUPLICATE_LIB_OK"):
            console.print("[yellow]Setting KMP_DUPLICATE_LIB_OK=TRUE[/yellow]")
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        # Import the actual processing components
        from ..config.unified_config import ModelType, UnifiedConfig
        from ..extraction.hybrid_extraction_manager import UnifiedExtractionManager

        console.print("[green]✅ Dependencies loaded successfully[/green]")

        # Validate model selection
        try:
            model_type = ModelType(model.lower())
        except ValueError:
            console.print(f"[red]❌ Invalid model: {model}[/red]")
            console.print(
                f"Available models: {', '.join([m.value for m in ModelType])}",
            )
            raise typer.Exit(1) from None

        # Create configuration
        config = UnifiedConfig.from_env()
        config.model_type = model_type

        # Process document using unified pipeline
        with console.status(f"[bold green]Processing with {model_type.value}..."):
            with UnifiedExtractionManager(config) as extraction_manager:
                result = extraction_manager.process_document(image_file)

        # Display results
        console.print("[green]✅ Processing completed successfully![/green]")
        console.print(f"Model: {result.model_type}")
        console.print(f"Document Type: {result.document_type}")
        console.print(f"Processing Time: {result.processing_time:.2f}s")
        console.print(f"Confidence Score: {result.confidence_score:.3f}")
        console.print(f"Quality Grade: {result.quality_grade.value}")

    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
        console.print("[yellow]Please ensure all dependencies are installed:[/yellow]")
        console.print("  conda activate unified_vision_processor")
        console.print("  pip install torch torchvision transformers")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red]❌ Processing failed: {e}[/red]")
        if verbose:
            import traceback

            console.print(traceback.format_exc())
        raise typer.Exit(1) from e


@app.command()
def test_env():
    """Test the environment and dependencies."""
    console.print("[bold blue]🧪 Testing Environment[/bold blue]")

    # Check KMP_DUPLICATE_LIB_OK
    kmp_setting = os.environ.get("KMP_DUPLICATE_LIB_OK", "Not set")
    console.print(f"KMP_DUPLICATE_LIB_OK: {kmp_setting}")

    # Test basic imports
    dependencies = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("typer", "Typer"),
        ("rich", "Rich"),
    ]

    all_good = True
    for module_name, display_name in dependencies:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            console.print(f"[green]✓ {display_name}: {version}[/green]")
        except ImportError as e:
            console.print(f"[red]✗ {display_name}: {e}[/red]")
            all_good = False

    # Test vision processor imports
    try:
        from ..config.unified_config import UnifiedConfig  # noqa: F401

        console.print("[green]✓ UnifiedConfig import[/green]")
    except ImportError as e:
        console.print(f"[red]✗ UnifiedConfig import: {e}[/red]")
        all_good = False

    if all_good:
        console.print("[green]🎉 All dependencies are working![/green]")
    else:
        console.print("[red]❌ Some dependencies are missing[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
