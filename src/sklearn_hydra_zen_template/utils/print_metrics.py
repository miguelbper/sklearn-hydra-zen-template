from rich.console import Console
from rich.table import Table

Metrics = dict[str, float]


def print_metrics(metrics: Metrics, prefix: str) -> None:
    """Pretty print metrics in a table format using Rich.

    Args:
        metrics: Dictionary of metric names and values
        prefix: Prefix to use in the title (e.g., 'Validation' or 'Test')
    """
    console = Console()
    table = Table()
    table.add_column(f"{prefix} metric", style="cyan")
    table.add_column("Value", style="magenta")

    for name, value in metrics.items():
        table.add_row(name, f"{value:.16f}")

    console.print(table)
