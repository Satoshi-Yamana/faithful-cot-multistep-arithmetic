import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def collect_result_files(paths):
    result_files = []
    for path in paths:
        if path.is_dir():
            result_files.extend(sorted(path.glob("result_*.json")))
            result_files.extend(sorted(path.glob("test_result_*.json")))
        else:
            result_files.append(path)

    unique_files = []
    seen = set()
    for path in result_files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_files.append(path)
    return unique_files


def load_records(file_paths, metric_name):
    records = []
    skipped = []

    for file_path in file_paths:
        with file_path.open() as f:
            payload = json.load(f)

        patch_config = payload.get("patch_config")
        if patch_config is None:
            skipped.append(file_path)
            continue

        if metric_name not in payload:
            raise KeyError(f"{file_path} does not contain metric '{metric_name}'")

        records.append(
            {
                "file_path": str(file_path),
                "layer": patch_config.get("patch_start_layer"),
                "position": patch_config.get("patch_start_position"),
                "metric": float(payload[metric_name]),
                "tag": payload.get("date", file_path.stem),
            }
        )

    return records, skipped


def build_grid(records, aggregate):
    grouped = defaultdict(list)
    for record in records:
        key = (record["layer"], record["position"])
        grouped[key].append(record["metric"])

    layers = sorted({layer for layer, _ in grouped})
    positions = sorted({position for _, position in grouped})

    grid = [[None for _ in positions] for _ in layers]
    layer_to_row = {layer: idx for idx, layer in enumerate(layers)}
    position_to_col = {position: idx for idx, position in enumerate(positions)}

    for (layer, position), values in grouped.items():
        if aggregate == "mean":
            value = float(mean(values))
        elif aggregate == "max":
            value = float(max(values))
        elif aggregate == "min":
            value = float(min(values))
        elif aggregate == "last":
            value = float(values[-1])
        else:
            raise ValueError(f"Unsupported aggregate: {aggregate}")

        grid[layer_to_row[layer]][position_to_col[position]] = value

    return layers, positions, grid


def render_heatmap(layers, positions, grid, title, metric_name, output_path, vmin, vmax, cmap):
    """Render heatmap using matplotlib and save to file (PNG or PDF)."""
    
    # Prepare data: convert grid to numpy array, replacing None with NaN
    data = np.array(grid, dtype=float)
    data[data == None] = np.nan
    
    # Invert layer axis for visualization (bottom-to-top)
    data = np.flipud(data)
    
    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create heatmap
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax, interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(positions)))
    ax.set_yticks(np.arange(len(layers)))
    ax.set_xticklabels(positions, rotation=45, ha='right')
    ax.set_yticklabels(list(reversed(layers)))
    
    # Labels and title
    ax.set_xlabel('Patch start position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Layer', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.set_xticks(np.arange(len(positions)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(layers)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label=metric_name, shrink=0.8)
    cbar.ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
    
    # Add text annotations on cells
    for i in range(len(layers)):
        for j in range(len(positions)):
            value = grid[len(layers) - 1 - i][j]
            if value is not None:
                text_color = 'black' if (vmin + vmax) / 2 > value else 'white'
                ax.text(j, i, f'{value:.2f}',
                       ha='center', va='center', color=text_color, fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if str(output_path).endswith('.png'):
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    else:
        # Default to PDF or PNG based on extension
        plt.savefig(output_path, bbox_inches='tight')
    
    plt.close()




def main():
    parser = argparse.ArgumentParser(description="Plot a causal intervention heatmap from result JSON files.")
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Input result JSON files or directories containing result_*.json files",
    )
    parser.add_argument(
        "--metric",
        default="intervention_accuracy",
        choices=["intervention_accuracy", "accuracy", "base_accuracy"],
        help="Metric to visualize",
    )
    parser.add_argument(
        "--aggregate",
        default="mean",
        choices=["mean", "max", "min", "last"],
        help="How to aggregate duplicate layer/position entries",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("heatmap.png"),
        help="Output image path (PNG, PDF, or SVG)",
    )
    parser.add_argument(
        "--title",
        default="Intervention Success Rate",
        help="Plot title",
    )
    parser.add_argument(
        "--cmap",
        default="RdPu",
        help="Matplotlib colormap name",
    )
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    args = parser.parse_args()

    input_files = collect_result_files(args.paths)
    records, skipped = load_records(input_files, args.metric)

    if not records:
        if skipped:
            skipped_paths = "\n".join(str(path) for path in skipped[:5])
            raise SystemExit(
                "No plottable result files were found. The selected inputs do not contain patch_config, "
                "so they cannot be converted into a layer-by-position heatmap.\n"
                f"Examples of skipped files:\n{skipped_paths}"
            )
        raise SystemExit("No result files found.")

    layers, positions, grid = build_grid(records, args.aggregate)
    render_heatmap(layers, positions, grid, args.title, args.metric, args.output, args.vmin, args.vmax, args.cmap)

    print(f"Saved heatmap to {args.output}")
    if skipped:
        print(f"Skipped {len(skipped)} file(s) without patch_config")


if __name__ == "__main__":
    main()