from rich.table import Table
from rich.console import Console
from io import StringIO

def log_results(aggregated_df, logger):
    console = Console(file=StringIO())
    table = Table(title="Aggregated Results")

    table.add_column("Num Sequence", justify="right", style="cyan")
    table.add_column("Metric", justify="left", style="magenta")
    table.add_column("Mean", justify="right", style="green")
    table.add_column("Std Error", justify="right", style="yellow")

    for _, row in aggregated_df.iterrows():
        table.add_row(
            str(row["num_sequence"]),
            row["metric"],
            f"{row['mean']:.4f}",
            f"{row['std_error']:.4f}",
        )

    with console.capture() as capture:
        console.print(table)

    return capture.get()

def generate_latex_row(aggregated_df, metrics_order):
    row_entries = []
    for metric in metrics_order:
        metric_data = aggregated_df[aggregated_df["metric"] == metric]
        if not metric_data.empty:
            mean = metric_data["mean"].values[0] * 100  # Convert to percentage
            std_error = (
                metric_data["std_error"].values[0] * 100
            )  # Convert to percentage
            row_entries.append(f"${mean:.2f} \pm {std_error:.2f}$")
        else:
            row_entries.append("N/A")

    return " & ".join(row_entries)