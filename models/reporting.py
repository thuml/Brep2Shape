import pathlib
from collections.abc import Mapping

from torch import nn


def _count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _append_module_tree(
    lines: list[str],
    module: nn.Module,
    *,
    indent: int = 2,
) -> None:
    for name, child in module.named_children():
        parameters = _count_parameters(child)
        if parameters == 0:
            continue
        prefix = " " * indent
        lines.append(f"{prefix}|- {name}: {parameters:>12,} ({parameters / 1e6:.2f}M)")
        _append_module_tree(lines, child, indent=indent + 2)


def format_parameter_report(components: Mapping[str, nn.Module]) -> str:
    """Format trainable parameter counts for named model components."""
    lines = ["", "=" * 80, "Detailed model parameter counts:", "=" * 80]
    total = 0
    for index, (name, module) in enumerate(components.items(), start=1):
        parameters = _count_parameters(module)
        total += parameters
        lines.append(f"\n[{index}] {name}: {parameters:>12,} ({parameters / 1e6:.2f}M)")
        _append_module_tree(lines, module)

    lines.extend(
        [
            "",
            "-" * 80,
            f"Total Parameters: {total:>12,} ({total / 1e6:.2f}M)",
            "=" * 80,
            "",
        ]
    )
    return "\n".join(lines)


def report_parameters(
    components: Mapping[str, nn.Module],
    save_path: str | pathlib.Path | None = None,
) -> str:
    """Print a parameter report and optionally write it to disk."""
    output = format_parameter_report(components)
    print(output)
    if save_path is not None:
        destination = pathlib.Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(output, encoding="utf-8")
        print(f"Parameter report saved to: {destination}")
    return output
