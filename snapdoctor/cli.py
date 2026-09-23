import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import json

console = Console()


@click.group()
def cli():
    """SnapDoctor — AI model diagnostics for Snapdragon NPU deployment."""
    pass


@cli.command()
@click.argument("model_path")
@click.argument("log_path", required=False, default=None)
def report(model_path, log_path):
    """Run static op-compatibility diagnosis on an ONNX model."""
    from snapdoctor.graph_parser import load_graph, extract_ops
    from snapdoctor.fallback_detector import diagnose, residency_summary, check_quantization_requirement
    from snapdoctor.qnn_log_parser import parse_log

    console.print(Panel(f"[bold cyan]Diagnosing:[/] {model_path}", box=box.ROUNDED))

    ops = extract_ops(load_graph(model_path))

    fallback_events = []
    if log_path:
        fallback_events = parse_log(log_path)

    diagnoses = diagnose(ops, fallback_events)
    summary = residency_summary(diagnoses)

    pct = summary["residency_pct"]
    color = "green" if pct > 90 else "yellow" if pct > 60 else "red"

    console.print(
        f"\n[bold]Op-level NPU compatibility:[/] [{color}]{pct}%[/] "
        f"({summary['resident_ops']}/{summary['total_ops']} ops)\n"
    )

    fallback_ops = summary["fallback_ops"]
    if fallback_ops:
        table = Table(title="Unsupported / Fallback Ops", box=box.SIMPLE)
        table.add_column("Op", style="red")
        table.add_column("Type")
        table.add_column("Reason")
        for d in fallback_ops[:15]:
            table.add_row(d.op_name, d.op_type, d.reason)
        console.print(table)
        if len(fallback_ops) > 15:
            console.print(f"[dim]... and {len(fallback_ops) - 15} more[/]")

    quant_info = check_quantization_requirement(model_path)
    quant_color = "green" if quant_info["htp_runnable"] else "red"
    console.print(f"\n[bold]HTP (NPU) runnable:[/] [{quant_color}]{quant_info['htp_runnable']}[/]")
    console.print(f"  {quant_info['note']}")


@cli.command()
@click.argument("profile_json")
def analyze(profile_json):
    """Show cycle-level bottleneck breakdown from an AI Hub profile."""
    with open(profile_json) as f:
        data = json.load(f)

    details = data.get("execution_detail", [])
    total_cycles = sum(d.get("execution_cycles", 0) for d in details)
    npu_ops = sum(1 for d in details if d.get("compute_unit") == "NPU")

    console.print(Panel(f"[bold cyan]Hardware Profile Analysis[/]", box=box.ROUNDED))
    if npu_ops == len(details):
        console.print(f"NPU residency: [green]{npu_ops}/{len(details)} ops (100%)[/]")
    else:
        console.print(f"[yellow]{npu_ops}/{len(details)} ops on NPU[/]")
    console.print(f"Total cycles: [bold]{total_cycles:,}[/]\n")

    from collections import defaultdict
    by_type = defaultdict(lambda: [0, 0])
    for d in details:
        t = d.get("type", "unknown")
        by_type[t][0] += d.get("execution_cycles", 0)
        by_type[t][1] += 1

    table = Table(title="Top Cost Contributors", box=box.SIMPLE)
    table.add_column("Node Type")
    table.add_column("Cycles", justify="right")
    table.add_column("% of Total", justify="right")
    table.add_column("Node Count", justify="right")

    for t, (cycles, count) in sorted(by_type.items(), key=lambda x: -x[1][0])[:10]:
        pct = cycles / total_cycles * 100 if total_cycles else 0
        table.add_row(t, f"{cycles:,}", f"{pct:.1f}%", str(count))

    console.print(table)


if __name__ == "__main__":
    cli()