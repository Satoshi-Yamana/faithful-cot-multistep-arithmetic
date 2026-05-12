"""
プロービング結果の可視化スクリプト
JSONLファイルからヒートマップと折れ線グラフを生成
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_probing_results(jsonl_path):
    """JSONLファイルを読み込んでデータを集計"""
    results = []
    with open(jsonl_path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def aggregate_scores(results):
    """
    結果を集計して、intermediate_answerごとに分類
    各intermediate_answerについて層×位置のグリッドを作成
    """
    # intermediate_answer -> (layer, position) -> score
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
        
        intermediate_answers.add(intermediate_answer)
        tokens[position] = token
        
        # intermediate_answerごとに分類
        grid_data_by_answer[intermediate_answer][layer][position] = score
        position_max_scores_by_answer[intermediate_answer][position] = score
    
    return grid_data_by_answer, position_max_scores_by_answer, tokens, intermediate_answers


def create_visualization(results, output_path, title=None):
    """
    プロービング結果を可視化
    intermediate_answerごとに異なる図を生成
    各図：上段は折れ線グラフ、下段はヒートマップ
    """
    grid_data_by_answer, position_max_scores_by_answer, tokens, intermediate_answers = aggregate_scores(results)
    
    # intermediate_answerの名前マッピング
    answer_names = {0: "Variable A", 1: "Variable B", 2: "Variable C"}
    
    # 各intermediate_answerごとに図を生成
    for answer_idx in sorted(intermediate_answers):
        grid_data = grid_data_by_answer[answer_idx]
        position_max_scores = position_max_scores_by_answer[answer_idx]
        
        # 層と位置を取得
        layers = sorted(grid_data.keys())
        if layers:
            positions = sorted(set(
                pos for layer_dict in grid_data.values() 
                for pos in layer_dict.keys()
            ))
        else:
            positions = []
        
        # ヒートマップ用グリッド作成
        if layers and positions:
            heatmap_grid = np.zeros((len(layers), len(positions)))
            for i, layer in enumerate(layers):
                for j, pos in enumerate(positions):
                    heatmap_grid[i, j] = grid_data[layer].get(pos, 0.0)
        else:
            heatmap_grid = np.array([])
        
        # 折れ線グラフ用データ
        line_positions = sorted(position_max_scores.keys())
        line_scores = [position_max_scores[pos] for pos in line_positions]
        
        # 図作成
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        answer_name = answer_names.get(answer_idx, f"Answer {answer_idx}")
        fig_title = title or f"Probing Results - {answer_name}"
        fig.suptitle(fig_title, fontsize=16, fontweight='bold')
        
        # 上段：折れ線グラフ
        ax_line = axes[0]
        ax_line.plot(line_positions, line_scores, marker='o', linewidth=2, markersize=4, color='purple')
        ax_line.set_ylabel('Probing accuracy', fontsize=12, fontweight='bold')
        ax_line.set_ylim([0, 1.0])
        ax_line.grid(True, alpha=0.3)
        ax_line.set_title(f'Max Probing Accuracy at Each Position ({answer_name})', fontsize=12)
        
        # 下段：ヒートマップ
        ax_heatmap = axes[1]
        if heatmap_grid.size > 0:
            im = ax_heatmap.imshow(
                heatmap_grid,
                aspect='auto',
                cmap='RdPu',
                vmin=0.0,
                vmax=1.0,
                origin='upper'
            )
            
            # x軸: 位置とトークン
            ax_heatmap.set_xticks(range(len(positions)))
            ax_heatmap.set_xticklabels([f"{p}" for p in positions], rotation=45, ha='right')
            
            # y軸: レイヤー
            ax_heatmap.set_yticks(range(len(layers)))
            ax_heatmap.set_yticklabels([f"Layer {l}" for l in layers])
            
            ax_heatmap.set_xlabel('Position (t)', fontsize=12, fontweight='bold')
            ax_heatmap.set_ylabel('Layer (l)', fontsize=12, fontweight='bold')
            ax_heatmap.set_title(f'Probing Accuracy Heatmap - {answer_name} (Layer × Position)', fontsize=12)
            
            # カラーバー
            cbar = plt.colorbar(im, ax=ax_heatmap)
            cbar.set_label('Accuracy', fontsize=11, fontweight='bold')
            
            # グリッド線
            ax_heatmap.set_xticks(np.arange(len(positions)) - 0.5, minor=True)
            ax_heatmap.set_yticks(np.arange(len(layers)) - 0.5, minor=True)
            ax_heatmap.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        # 出力パスに答えのインデックスを挿入
        output_file = output_path.parent / f"{output_path.stem}_answer_{answer_idx}{output_path.suffix}"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ {answer_name} の図を保存しました: {output_file}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="プロービング結果をJSONLから可視化"
    )
    parser.add_argument(
        "input_jsonl",
        type=str,
        help="入力JSONLファイルパス (e.g., linear_classifier_result_*.jsonl)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="出力画像パス (デフォルト: 入力ファイル名から.pngに変更)"
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="グラフのタイトル"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        print(f"✗ ファイルが見つかりません: {input_path}")
        return
    
    # 出力パスを決定
    if args.output is None:
        # デフォルトは result_probing ディレクトリに保存
        result_probing_dir = Path("result_probing")
        result_probing_dir.mkdir(exist_ok=True, parents=True)
        output_path = result_probing_dir / f"{input_path.stem}.png"
    else:
        output_path = Path(args.output)
    
    print(f"📖 読み込み中: {input_path}")
    results = load_probing_results(input_path)
    print(f"   {len(results)} 件のレコードを読み込みました")
    
    title = args.title or f"Probing Results ({input_path.stem})"
    
    print(f"🎨 可視化を生成中...")
    create_visualization(results, output_path, title=title)


if __name__ == "__main__":
    main()
