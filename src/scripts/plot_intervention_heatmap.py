import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable


def collect_result_files(paths: List[Path]) -> List[Path]:
    result_files = []
    for path in paths:
        if path.is_dir():
            result_files.extend(path.glob("result_*.json"))
            result_files.extend(path.glob("test_result_*.json"))
        else:
            result_files.append(path)

    # Resolve paths to remove duplicates
    return list({p.resolve(): p for p in result_files}.values())


def load_token_mapping(jsonl_path: Path) -> Dict[int, str]:
    pos_to_token = {}
    if not jsonl_path or not jsonl_path.exists():
        return pos_to_token
        
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            pos = data.get('position')
            token = data.get('token', '')
            
            if pos is not None and pos not in pos_to_token:
                # Replace 'Ġ' with a standard space and escape newlines
                token_str = str(token).replace('Ġ', ' ').replace('\n', '\\n')
                pos_to_token[pos] = token_str
                
    return pos_to_token


def load_records(file_paths: List[Path], metric_name: str) -> Tuple[pd.DataFrame, List[Path]]:
    records = []
    skipped = []

    for file_path in file_paths:
        with file_path.open('r', encoding='utf-8') as f:
            try:
                payload = json.load(f)
            except json.JSONDecodeError:
                skipped.append(file_path)
                continue

        patch_config = payload.get("patch_config")
        if patch_config is None:
            skipped.append(file_path)
            continue

        if metric_name not in payload:
            raise KeyError(f"{file_path} does not contain metric '{metric_name}'")

        # ブロック全体を塗れるように end_layer と end_position も取得するよ！
        # もし値がなかったら、開始位置と同じにして1マスのブロックとして扱うね
        start_layer = patch_config.get("patch_start_layer")
        start_pos = patch_config.get("patch_start_position")
        
        records.append({
            "layer": start_layer,
            "position": start_pos,
            "end_layer": patch_config.get("patch_end_layer", start_layer),
            "end_position": patch_config.get("patch_end_position", start_pos),
            "metric": float(payload[metric_name]),
        })

    return pd.DataFrame(records), skipped


def render_heatmap(
    df: pd.DataFrame,
    pos_to_token: Dict[int, str],
    aggregate: str,
    title: str,
    metric_name: str,
    output_path: Path,
    vmin: float,
    vmax: float,
    cmap: str
):
    # 0から最大のposition/layerまで全て補完して表示するよ！
    if not df.empty:
        max_pos = int(df[["position", "end_position"]].max().max())
        max_layer = int(df[["layer", "end_layer"]].max().max())
        positions = list(range(0, max_pos + 1))
        layers = list(range(0, max_layer + 1))
    else:
        positions = []
        layers = []

    # ブロックごとに値を集計するよ！
    grid_values = defaultdict(list)
    
    for _, row in df.iterrows():
        l_start, l_end = int(row['layer']), int(row['end_layer'])
        p_start, p_end = int(row['position']), int(row['end_position'])
        val = row['metric']
        
        # ブロック内のすべてのセルに値をセットするね
        for l in range(l_start, l_end + 1):
            for p in range(p_start, p_end + 1):
                grid_values[(l, p)].append(val)

    # ヒートマップ用のNumPy配列を作成
    heatmap_grid = np.zeros((len(layers), len(positions)))
    
    for (l, p), vals in grid_values.items():
        if l in layers and p in positions:
            l_idx = layers.index(l)
            p_idx = positions.index(p)
            
            # 同じマスに複数回パッチが当たった場合の集計方法だよ
            if aggregate == 'mean':
                heatmap_grid[l_idx, p_idx] = np.mean(vals)
            elif aggregate == 'max':
                heatmap_grid[l_idx, p_idx] = np.max(vals)
            elif aggregate == 'min':
                heatmap_grid[l_idx, p_idx] = np.min(vals)
            elif aggregate == 'last':
                heatmap_grid[l_idx, p_idx] = vals[-1]

    # トークンの数に合わせて横幅を長めに調整するね
    fig_width = max(14, len(positions) * 0.4)
    fig_height = max(5, len(layers) * 0.4)
    
    fig, ax_heatmap = plt.subplots(figsize=(fig_width, fig_height))
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

    im = ax_heatmap.imshow(
        heatmap_grid,
        aspect='auto',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin='lower'
    )

    x_indices = np.arange(len(positions))
    x_lim = [-0.5, len(positions) - 0.5]
    
    x_labels = []
    for p in positions:
        token_str = pos_to_token.get(p, "")
        if token_str:
            x_labels.append(f"{p}\n{token_str}")
        else:
            x_labels.append(f"{p}")

    ax_heatmap.set_xlim(x_lim)
    ax_heatmap.set_xticks(x_indices)
    ax_heatmap.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=8)
    ax_heatmap.set_xlabel('Position (t) & Token', fontsize=11, fontweight='bold')

    ax_heatmap.set_yticks(range(len(layers)))
    ax_heatmap.set_yticklabels([f"L{int(l)}" for l in layers], fontsize=8)
    ax_heatmap.set_ylabel('Layer (l)', fontsize=11, fontweight='bold')
    
    # パッチの範囲を薄いグレーの実線ブロックで表示するよ！
    unique_blocks = df[['position', 'layer', 'end_position', 'end_layer']].drop_duplicates()
    
    for _, row in unique_blocks.iterrows():
        pos, layer = int(row['position']), int(row['layer'])
        end_pos, end_layer = int(row['end_position']), int(row['end_layer'])
        
        if pos in positions and layer in layers and end_pos in positions and end_layer in layers:
            x_idx = positions.index(pos)
            y_idx = layers.index(layer)
            end_x_idx = positions.index(end_pos)
            end_y_idx = layers.index(end_layer)
            
            # ブロックの幅と高さを計算するよ
            pos_width = (end_x_idx - x_idx) + 1
            layer_height = (end_y_idx - y_idx) + 1
            
            rect = patches.Rectangle(
                (x_idx - 0.5, y_idx - 0.5),
                pos_width,
                layer_height,
                linewidth=1.5,
                edgecolor='lightgray',
                facecolor='none',
                linestyle='-' # 実線
            )
            ax_heatmap.add_patch(rect)

    # カラーバー
    divider_heatmap = make_axes_locatable(ax_heatmap)
    cax_heatmap = divider_heatmap.append_axes("right", size="2%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax_heatmap)
    cbar.set_label(metric_name, fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # グリッド
    ax_heatmap.set_xticks(np.arange(len(positions)) - 0.5, minor=True)
    ax_heatmap.set_yticks(np.arange(len(layers)) - 0.5, minor=True)
    ax_heatmap.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
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
        "--token-file",
        type=Path,
        default=None,
        help="Path to the JSONL file mapping position to token",
    )
    parser.add_argument(
        "--metric",
        default="intervention_accuracy",
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
        default=Path("heatmap_clean.png"),
        help="Output image path",
    )
    parser.add_argument(
        "--title",
        default="Intervention Success Rate",
        help="Plot title",
    )
    parser.add_argument(
        "--cmap",
        default="Blues",
        help="Matplotlib colormap name",
    )
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    args = parser.parse_args()

    input_files = collect_result_files(args.paths)
    df, skipped = load_records(input_files, args.metric)

    if df.empty:
        if skipped:
            skipped_paths = "\n".join(str(path) for path in skipped[:5])
            raise SystemExit(
                "No plottable result files were found.\n"
                f"Examples of skipped files:\n{skipped_paths}"
            )
        raise SystemExit("No result files found.")

    pos_to_token = load_token_mapping(args.token_file)
    
    render_heatmap(
        df=df,
        pos_to_token=pos_to_token,
        aggregate=args.aggregate,
        title=args.title,
        metric_name=args.metric,
        output_path=args.output,
        vmin=args.vmin,
        vmax=args.vmax,
        cmap=args.cmap
    )

    print(f"Saved cleanly formatted heatmap to {args.output}")
    if skipped:
        print(f"Skipped {len(skipped)} file(s) without patch_config")


if __name__ == "__main__":
    main()