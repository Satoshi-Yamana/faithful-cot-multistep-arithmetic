"""
プロービング結果の統合可視化スクリプト
- タイトル指定
- 変数ごとの色とシンボルの出し分け
- 横軸のトークン表示と位置合わせ
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def load_probing_results(jsonl_path):
    """JSONLファイルを読み込んでデータを集計"""
    results = []
    with open(jsonl_path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def aggregate_scores(results):
    """結果を集計して、intermediate_answerごとに分類"""
    grid_data_by_answer = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    position_max_scores_by_answer = defaultdict(lambda: defaultdict(float))
    tokens = {}  # position -> token
    intermediate_answers = set()
    
    for record in results:
        layer = record["layer"]
        position = record["position"]
        score = record["score"]
        token = record["token"]
        intermediate_answer = record["intermediate_answer"]
        
        # 空白文字 'Ġ' や改行を見やすく置換
        clean_token = token.replace('Ġ', ' ').replace('\n', '\\n')
        
        intermediate_answers.add(intermediate_answer)
        tokens[position] = clean_token
        
        grid_data_by_answer[intermediate_answer][layer][position] = score
        position_max_scores_by_answer[intermediate_answer][position] = max(
            position_max_scores_by_answer[intermediate_answer][position], score
        )
    
    return grid_data_by_answer, position_max_scores_by_answer, tokens, intermediate_answers


def create_visualization_combined(results, output_path, title=None):
    grid_data_by_answer, position_max_scores_by_answer, tokens, intermediate_answers = aggregate_scores(results)
    
    # ユーザー指定の順序: Z(1)が上, O(0)が下
    sorted_answers = [1, 0]  # Zが上、Oが下の順序
    
    # 設定：名前、色、マーカー、カラーマップの対応付け
    answer_config = {
        0: {"name": "Variable O", "color": "darkorange", "marker": "o", "cmap": "Oranges"},
        1: {"name": "Variable Z", "color": "seagreen", "marker": "^", "cmap": "Greens"}
    }

    # 全データを通しての層と位置を取得
    all_layers = set()
    all_positions = set()
    for ans in sorted_answers:
        if ans in grid_data_by_answer:
            all_layers.update(grid_data_by_answer[ans].keys())
            for l in grid_data_by_answer[ans]:
                all_positions.update(grid_data_by_answer[ans][l].keys())
            
    layers = sorted(list(all_layers))
    positions = sorted(list(all_positions))
    num_answers = len(sorted_answers)

    # グラフ作成
    height_ratios = [1.5] + [2.5] * num_answers
    fig, axes = plt.subplots(num_answers + 1, 1, figsize=(14, 4 + 4 * num_answers), 
                             gridspec_kw={'height_ratios': height_ratios})
    
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # 指定のタイトルを設定
    fig_title = title or "Probing results for Qwen2.5-7B at the Level 3 task"
    fig.suptitle(fig_title, fontsize=16, fontweight='bold', y=0.98)

    # 共通のX軸設定
    x_indices = np.arange(len(positions))
    x_lim = [-0.5, len(positions) - 0.5]
    x_labels = [f"{p}\n{tokens.get(p, '')}" for p in positions]

    # ==========================================
    # 1. 上段：折れ線グラフ (シンボルと色を分ける)
    # ==========================================
    ax_line = axes[0]
    for answer_idx in sorted_answers:
        if answer_idx not in position_max_scores_by_answer: continue
        
        max_scores = position_max_scores_by_answer[answer_idx]
        line_scores = [max_scores.get(pos, np.nan) for pos in positions]
        conf = answer_config.get(answer_idx)
        
        ax_line.plot(
            x_indices, line_scores,
            label=conf["name"],
            marker=conf["marker"], # 異なるシンボル
            linewidth=2, 
            markersize=5, 
            color=conf["color"],   # 揃えた色
            alpha=0.9
        )
    
    ax_line.set_xlim(x_lim)
    ax_line.set_xticks(x_indices)
    ax_line.tick_params(labelbottom=False) 
    ax_line.set_ylabel('Max Accuracy', fontsize=11, fontweight='bold')
    ax_line.set_ylim([0, 1.05])
    ax_line.grid(True, linestyle='--', alpha=0.3)
    ax_line.legend(loc='upper left', fontsize=10, frameon=True)

    # ヒートマップと横幅を揃えるための透明なスペース
    divider_line = make_axes_locatable(ax_line)
    cax_dummy = divider_line.append_axes("right", size="3%", pad=0.1)
    cax_dummy.axis('off')

    # ==========================================
    # 2. 下段以降：ヒートマップ (Zが上, Oが下)
    # ==========================================
    for i, answer_idx in enumerate(sorted_answers):
        ax_heatmap = axes[i + 1]
        conf = answer_config.get(answer_idx)
        grid_data = grid_data_by_answer[answer_idx]

        heatmap_grid = np.zeros((len(layers), len(positions)))
        for row_idx, layer in enumerate(layers):
            for col_idx, pos in enumerate(positions):
                heatmap_grid[row_idx, col_idx] = grid_data[layer].get(pos, 0.0)

        im = ax_heatmap.imshow(
            heatmap_grid,
            aspect='auto',
            cmap=conf["cmap"], # 各変数に合わせたカラーマップ
            vmin=0.0,
            vmax=1.0,
            origin='lower'
        )

        ax_heatmap.set_xlim(x_lim)
        ax_heatmap.set_xticks(x_indices)
        
        # 最後の段にだけトークンラベルを表示
        if i == num_answers - 1:
            ax_heatmap.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=8)
            ax_heatmap.set_xlabel('Position (t) & Token', fontsize=11, fontweight='bold')
        else:
            ax_heatmap.tick_params(labelbottom=False)

        ax_heatmap.set_yticks(range(len(layers)))
        ax_heatmap.set_yticklabels([f"L{l}" for l in layers], fontsize=8)
        ax_heatmap.set_ylabel('Layer (l)', fontsize=11, fontweight='bold')
        ax_heatmap.set_title(f'Probing Accuracy Heatmap: {conf["name"]}', fontsize=12, color=conf["color"], fontweight='bold')

        # カラーバー
        divider_heatmap = make_axes_locatable(ax_heatmap)
        cax_heatmap = divider_heatmap.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax_heatmap)
        cbar.ax.tick_params(labelsize=8)

        # グリッド
        ax_heatmap.set_xticks(np.arange(len(positions)) - 0.5, minor=True)
        ax_heatmap.set_yticks(np.arange(len(layers)) - 0.5, minor=True)
        ax_heatmap.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.35)

    output_file = output_path.parent / f"final_probing_visualization.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ 可視化を保存しました: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", type=str)
    args = parser.parse_args()
    
    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        print(f"✗ ファイルがありません: {input_path}")
        return
    
    print(f"📖 処理中: {input_path.name}")
    results = load_probing_results(input_path)
    
    # 指示通りのタイトルを渡す
    create_visualization_combined(
        results, 
        input_path, 
        title="Probing results for Qwen2.5-7B at the Level 3 task"
    )

if __name__ == "__main__":
    main()