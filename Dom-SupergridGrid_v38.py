8
# =========================================================
#  Dom-SupergridGrid v38_GCN-GATv2-SAGE-GIN-TRANSFORMER-GRAPHORMER-GPSCONV
#  Supergrid / Grid Domination
#
#  • Method:
#          GCN-GATv2-SAGE-GIN-TRANSFORMER-GRAPHORMER-GPSCONV
#
#  • 不規則 Grid / Supergrid
#       隨機挖洞-hole rate setting
#       挖洞後需為連通圖; 若挖洞多次都不連通, 則停止挖洞並輸出挖洞例
#  • ILP + Greedy dual-teachers Training: ILP=0.5 & Greedy=0.5
#  
#  • Base Node Features + Laplacian PE + Random-Walk Embedding (RWE)
#  • AMP + Gradient Clipping + Curriculum (小圖→大圖)  (簡化實作)
#  • Sequential Actor–Critic RL fine-tuning + Entropy Regularization (簡化實作)
#  
# GNN 補點:
#    (1) GNN 產生 raw (thredhold = 0.5)
#    (2) prune
#    (3) 若 coverage 不等於 1，先對「未被 cover 的點」做 ILP 補點
#    (4) 若還沒補滿，再用 beam search
#    (5) 若還沒補滿，最後才用 guided greedy
#    (6) 最後再做 prune + local swap

# RL fine-tuning 修正:
#    (1) 所有 GNN 共用的 train_actor_critic_for_model() 都改成較省顯存版本
#        不再把整個 episode 的很多圖一起累積到最後才 opt.step()
#        改成 每張圖即時 backward / step / 清 cache
#    (2) 所有 GNN 的 RL rollout 都限制步數
#        加入 RL_MAX_STEPS_PER_GRAPH = 64
#    (3) 所有 GNN 的 RL 都有限制可進入的圖大小
#        全域 RL_MAX_GRAPH_N_FOR_RL = 700
#        GPSCONV 額外更保守 RL_MAX_GRAPH_N_FOR_RL_GPSCONV = 450
#    (4) 原本就有的 generic RL 流程，現在真的對全部模型一致套用
#        因為 train_actor_critic_for_model() 本來就是寫成給 GCN / GATv2 / SAGE / GIN / TRANSFORMER / GRAPHORMER / GPSCONV 共用。
#    (5) GPSCONV 另外保留較小模型設定
#        這是因為它本來就比其他 GNN 更容易炸顯存。

# ILP 修正: (ILP可能沒能解出Opt - feas_nonopt / ilp_fail_missing - time out or failure)
#    ILP: OPTIMAL / TIME_LIMIT_FEASIBLE / FAILED (RL Train + Test Data-Func 14 + 3Methods-Func 12)

import os
import sys
import math
import random
import time
import copy
import itertools
import datetime
from typing import List, Tuple, Dict, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from torch.cuda.amp import autocast, GradScaler

from torch_geometric.data import Data

from torch_geometric.nn import (
    GCNConv,
    GATv2Conv,
    SAGEConv,
    GINConv,
    TransformerConv,
    GPSConv,
    GINEConv, #GIN with edge features
    global_mean_pool,
)

import matplotlib.pyplot as plt

# =========================================================
#  Base Experiment Paths（統一管理）
# =========================================================
# Windows 會遇到「路徑太長」(MAX_PATH) 的問題；建議把資料輸出到短路徑。
# 你可用環境變數 DOM_BASE_DIR 覆蓋預設路徑，例如：DOM_BASE_DIR=D:/GNNDom
from pathlib import Path
import os
BASE_DIR = Path(__file__).resolve().parent

TRAINPATH  = BASE_DIR / "1TrainSet"
MODELPATH  = BASE_DIR / "2Models"
LOGPATH    = BASE_DIR / "3Logs"
RESULTPATH = BASE_DIR / "4Results"
TESTPATH   = BASE_DIR / "0TestSet"
# 你原程式大量使用 os.path.join / os.makedirs（偏字串），因此這裡保留 str 版本
EXPERIMENT_RESULTS_DIR = str(RESULTPATH)

# =========================================================
#  訓練參數
# =========================================================
TRAIN_DEFAULT_NUM = 1200
TRAIN_DEFAULT_MIN_M = 50
TRAIN_DEFAULT_MAX_M = 100
TRAIN_DEFAULT_MIN_N = 50
TRAIN_DEFAULT_MAX_N = 100
TRAIN_DEFAULT_HOLE_RATIO = 0.20  # 建議多次執行本功能，分別用 0.0,0.1,0.2,0.3,0.4, 0.5

# =========================================================
#  測試參數
# =========================================================
TEST_DEFAULT_NUM = 200
TEST_DEFAULT_MIN_M = 20
TEST_DEFAULT_MAX_M = 70
TEST_DEFAULT_MIN_N = 20
TEST_DEFAULT_MAX_N = 70
TEST_DEFAULT_HOLE_RATIO = 0.20

# =========================================================
#  Global 狀態
# =========================================================
GLOBAL_MODELS = {}
GLOBAL_DEVICE = None
GLOBAL_GRAPHS = None

GLOBAL_MODEL_IN_DIM = None

PRINT_PIPELINE_LOG = False

# =========================================================
#  Dual-teacher mixing weights (ILP / Greedy)
# =========================================================
ILP_WEIGHT = 0.5
GREEDY_WEIGHT = 0.5
SMALL_GRAPH_NODE_THRESHOLD = 400
SMALL_GRAPH_ILP_WEIGHT = 0.9
SMALL_GRAPH_GREEDY_WEIGHT = 0.1
DATASET_ILP_TIMEOUT = 30
SYMMETRY_AUGMENTATION = True
SYMMETRY_SWAP_RC_PROB = 0.5
SYMMETRY_FLIP_ROW_PROB = 0.5
SYMMETRY_FLIP_COL_PROB = 0.5

# =========================================================
#  Model Weight Naming Helpers（存檔/載入都以 GNN 名稱為準）
# =========================================================
# =========================================================
#  Train/Test Dataset File Naming（依 GNN 名稱）
# =========================================================
# =========================================================
#  GNN Selection（只訓練 / 載入 / 測試「一個」模型）
# =========================================================
AVAILABLE_GNNS = ["GCN", "GATv2", "SAGE", "GIN", "TRANSFORMER", "GRAPHORMER", "GPSCONV"]
SELECTED_GNN = "GCN"   # default


# =========================================================
#  Plot: nodes/edges + dominating set (paper-style)
#  - axes with 1-based coordinates, top-left is (1,1)
#  - hollow nodes, blue filled dominating nodes
#  - black edges
# =========================================================
def plot_methods_nodes_edges(
        adj,
        coords,
        result_sets,
        main_title=None,
        save_path=None,
        m=None,
        n=None,
        hole_rate=None,
        show_grid=True,
        show_all_methods=True,
):
    """
    adj        : list[set] adjacency (0..N-1)
    coords     : list[(row, col)] in 0-based
    result_sets: dict {method_name: list[int] dominating set indices}
    """

    # infer m,n if not provided
    if m is None:
        m = max((r for r, _ in coords), default=0) + 1
    if n is None:
        n = max((c for _, c in coords), default=0) + 1

    methods = list(result_sets.items())
    if not show_all_methods and methods:
        methods = methods[:1]

    k = max(len(methods), 1)
    fig, axes = plt.subplots(1, k, figsize=(5 * k, 5))
    if k == 1:
        axes = [axes]

    # ticks step (avoid too dense labels)
    x_step = 1 if n <= 30 else (2 if n <= 60 else 5)
    y_step = 1 if m <= 30 else (2 if m <= 60 else 5)

    for ax, (name, D) in zip(axes, methods):
        Dset = set(D)

        # --- edges (black) ---
        for v in range(len(adj)):
            r1, c1 = coords[v]
            x1, y1 = c1 + 1, r1 + 1
            for u in adj[v]:
                if u > v:
                    r2, c2 = coords[u]
                    x2, y2 = c2 + 1, r2 + 1
                    ax.plot([x1, x2], [y1, y2], color="black", linewidth=0.6, zorder=1)

        # --- nodes (hollow) ---
        xs = [c + 1 for (r, c) in coords]
        ys = [r + 1 for (r, c) in coords]
        ax.scatter(xs, ys, s=45, facecolors="white", edgecolors="black", linewidths=1.0, zorder=2)

        # --- dominating nodes (blue filled) ---
        if Dset:
            dx = [coords[v][1] + 1 for v in Dset]
            dy = [coords[v][0] + 1 for v in Dset]
            ax.scatter(dx, dy, s=55, c="blue", edgecolors="blue", zorder=3)

        # --- axes: (1,1) at top-left ---
        ax.set_xlim(0.5, n + 0.5)
        ax.set_ylim(m + 0.5, 0.5)  # invert y-axis
        ax.set_xticks(list(range(1, n + 1, x_step)))
        ax.set_yticks(list(range(1, m + 1, y_step)))
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if show_grid:
            ax.grid(True, linestyle="--", alpha=0.3)

        ax.set_aspect("equal")

        # subplot title shows solution size
        if hole_rate is None:
            ax.set_title(f"{name}  |D|={len(D)}", fontsize=10)
        else:
            ax.set_title(f"{name}  |D|={len(D)}  {m}×{n}  hole={float(hole_rate):.2f}", fontsize=10)

    if main_title:
        fig.suptitle(main_title)

    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path, dpi=300)
            print(f"[Plot] saved -> {save_path}")
        except Exception as e:
            print(f"[Plot] save failed: {e}")

    plt.show()
from pathlib import Path

from pathlib import Path
import os

for _p in [TRAINPATH, MODELPATH, LOGPATH, RESULTPATH, TESTPATH]:
    _p.mkdir(parents=True, exist_ok=True)


# =========================================================
#  Folder Scan Helpers
# =========================================================
def scan_train_safe_Connected_folders(root=None):
    #在 TRAINPATH 下，找到所有 Train_* 資料夾（或指定 root）。
    if root is None:
        root = TRAINPATH
    root = str(root)
    if not os.path.isdir(root):
        return []
    folders = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and d.startswith("Train_")
    ]
    return sorted(folders)

def scan_testset_folders(root=None):
    #在 TESTPATH 下，找到所有 Test_* 資料夾（或指定 root）。
    if root is None:
        root = TESTPATH
    root = str(root)
    if not os.path.isdir(root):
        return []
    folders = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and d.startswith("Test_")
    ]
    return sorted(folders)


def choose_one_gnn():
    global SELECTED_GNN
    print("=== 選擇要執行的 GNN 模型 ===")
    for i, name in enumerate(AVAILABLE_GNNS, 1):
        print(f"{i}. {name}")
    s = input(f"請選擇 (1-{len(AVAILABLE_GNNS)}) 或直接輸入名稱 [目前 {SELECTED_GNN}]：").strip()
    if s == "":
        print(f"✔ 維持：{SELECTED_GNN}")
        return SELECTED_GNN
    if s.isdigit():
        k = int(s)
        if 1 <= k <= len(AVAILABLE_GNNS):
            SELECTED_GNN = AVAILABLE_GNNS[k-1]
            print(f"✔ 已選擇：{SELECTED_GNN}")
            return SELECTED_GNN
    s_up = s.upper()
    # allow flexible match
    for name in AVAILABLE_GNNS:
        if s_up == name.upper():
            SELECTED_GNN = name
            print(f"✔ 已選擇：{SELECTED_GNN}")
            return SELECTED_GNN
    print("❌ 無效選擇，維持原設定。")
    return SELECTED_GNN


def normalize_gnn_name(raw: str) -> str:
    """Normalize a raw name (from filename prefix / user input) into AVAILABLE_GNNS spelling."""
    if raw is None:
        return SELECTED_GNN
    s = str(raw).strip()
    if not s:
        return SELECTED_GNN
    s_up = s.upper().replace("_", "").replace("-", "")
    # common aliases
    mapping = {
        "GCN": "GCN",
        "GATV2": "GATv2",
        "GATV2CONV": "GATv2",
        "SAGE": "SAGE",
        "GRAPHSAGE": "SAGE",
        "GIN": "GIN",
        "GINE": "GIN",  # not used here, but keep a safe default
        "TRANSFORMER": "TRANSFORMER",
        "TRANSFORMERCONV": "TRANSFORMER",
        "GRAPHORMER": "GRAPHORMER",
        "GPS": "GPSCONV",
        "GPSCONV": "GPSCONV",
    }
    if s_up in mapping:
        return mapping[s_up]
    # fallback: try match AVAILABLE_GNNS ignoring case
    for name in AVAILABLE_GNNS:
        if s_up == name.upper().replace("_", "").replace("-", ""):
            return name
    return s  # last resort


def infer_in_dim_from_state_dict(model_name: str, state_dict: dict) -> int:
    """Infer model input feature dimension from a checkpoint state_dict."""
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Empty state_dict; cannot infer in_dim")

    # model-specific common keys
    if model_name == "GRAPHORMER":
        k = "lin_in.weight"
        if k in state_dict and getattr(state_dict[k], "ndim", 0) == 2:
            return int(state_dict[k].shape[1])
    if model_name == "GPSCONV":
        k = "node_encoder.weight"
        if k in state_dict and getattr(state_dict[k], "ndim", 0) == 2:
            return int(state_dict[k].shape[1])
    if model_name in ("GCN", "GIN"):
        k = "convs.0.lin.weight"
        if k in state_dict and getattr(state_dict[k], "ndim", 0) == 2:
            return int(state_dict[k].shape[1])
    if model_name == "SAGE":
        # SAGEConv has lin_l / lin_r
        for k in ("convs.0.lin_l.weight", "convs.0.lin_r.weight"):
            if k in state_dict and getattr(state_dict[k], "ndim", 0) == 2:
                return int(state_dict[k].shape[1])
    if model_name in ("GATv2", "TRANSFORMER"):
        # both have an initial linear projection
        for k in (
            "convs.0.lin_l.weight",
            "convs.0.lin_r.weight",
            "convs.0.lin_key.weight",
            "convs.0.lin_query.weight",
            "convs.0.lin_value.weight",
        ):
            if k in state_dict and getattr(state_dict[k], "ndim", 0) == 2:
                return int(state_dict[k].shape[1])

    # generic fallback: choose the smallest plausible input dim from any 2D weight
    candidates = []
    for k, v in state_dict.items():
        if getattr(v, "ndim", 0) == 2:
            candidates.append(int(v.shape[1]))
    if not candidates:
        raise ValueError("No 2D weight found in state_dict; cannot infer in_dim")
    # prefer the minimum (often corresponds to input features, not hidden dim)
    return int(min(candidates))


from pathlib import Path
# =========================================================
#  Global Settings
# =========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
print("[INFO] Using device:", DEVICE)


# ================= v32 CAMERA READY SYSTEM =================
# 🔥 Final: publication-quality plots + anomaly detection + captions

import csv
import time
import math

# ================= VRAM =================
class VRAMAdaptiveController:
    def __init__(self, min_batch=1, max_batch=12):
        self.batch_size = min_batch
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.success_steps = 0

    def success(self):
        self.success_steps += 1
        if self.success_steps >= 2:
            self.batch_size = min(self.batch_size + 1, self.max_batch)
            self.success_steps = 0

    def fail(self):
        self.batch_size = max(self.batch_size // 2, self.min_batch)
        self.success_steps = 0

    def get(self):
        return self.batch_size


GLOBAL_VRAM_CONTROLLER = VRAMAdaptiveController()


# ================= EXP =================
def run_experiment(models, graph_generator, sizes, hole_rates):
    results = []
    for (m, n) in sizes:
        for hole in hole_rates:
            adj, coords, _ = graph_generator(m, n, hole)
            for name, method in models.items():
                t0 = time.time()
                D = method(adj, coords)
                t = time.time() - t0
                results.append({
                    "model": name,
                    "m": m, "n": n,
                    "hole": hole,
                    "|D|": len(D),
                    "time_sec": t
                })
    return results


# ================= ANOMALY DETECTION =================
def detect_anomalies(results):
    issues = []
    times = [r["time_sec"] for r in results]
    mean = sum(times) / len(times)
    std = (sum((t-mean)**2 for t in times)/len(times))**0.5

    for r in results:
        if r["time_sec"] > mean + 2*std:
            issues.append(("slow", r))

    return issues


# ================= CSV =================
def save_results_csv(results, path="results.csv"):
    keys = results[0].keys()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


# ================= LATEX =================
def export_latex_table(results, path="table.tex"):
    with open(path, "w") as f:
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("Model & m & n & hole & |D| & time\\\\\n\\hline\n")
        for r in results:
            f.write(f"{r['model']} & {r['m']} & {r['n']} & {r['hole']} & {r['|D|']} & {r['time_sec']:.3f}\\\\\n")
        f.write("\\end{tabular}")


# ================= PLOT (Camera Ready) =================
def plot_camera_ready(results):
    import matplotlib.pyplot as plt

    models = sorted(set(r["model"] for r in results))
    plt.figure()

    for model in models:
        xs, ys = [], []
        for r in results:
            if r["model"] == model:
                xs.append(r["m"] * r["n"])
                ys.append(r["time_sec"])
        plt.plot(xs, ys, marker="o", label=model)

    plt.yscale("log")
    plt.xlabel("Graph Size (mn)")
    plt.ylabel("Time (log scale)")
    plt.title("Eight Methods Comparison (Camera Ready)")
    plt.legend()
    plt.grid(True)

    # inset zoom
    ax = plt.gca()
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax, width="40%", height="30%", loc="upper left")

    for model in models:
        xs, ys = [], []
        for r in results:
            if r["model"] == model:
                xs.append(r["m"] * r["n"])
                ys.append(r["time_sec"])
        axins.plot(xs, ys)

    axins.set_xlim(min(xs), min(xs)*2)
    axins.set_ylim(min(ys), sorted(ys)[len(ys)//3])
    axins.set_yscale("log")

    plt.savefig("camera_ready_figure.pdf", bbox_inches="tight")
    plt.show()


# ================= CAPTION =================
def generate_caption():
    return ("Comparison of eight methods on rectangular supergrid graphs. "
            "The y-axis is shown in logarithmic scale. The inset highlights "
            "performance differences on small-sized graphs.")


# =====================================================



GRAPH_TOPOLOGY = "supergrid"  # "grid" or "supergrid"

USE_RWE = True
RWE_DIM = 16

ILP_MaxTime = 100 # 100sec

ILP_CompleteTime = 60

# Prune+GuidedGreedy soft repair (GNN-guided exact repair)
PROB_SOFTREPAIR_THRESHOLD = 0.35 # GNN選進raw的threshold; 預設=0.5
PROB_SOFTREPAIR_USE_ADAPTIVE_THRESHOLD = True
PROB_SOFTREPAIR_TARGET_RAW_COVERAGE = 0.88
PROB_SOFTREPAIR_MIN_RAW_RATIO = 0.03
PROB_SOFTREPAIR_MAX_RAW_RATIO = 0.22
PROB_SOFTREPAIR_ADAPTIVE_GAIN_WEIGHT = 1.0
PROB_SOFTREPAIR_ADAPTIVE_SCORE_WEIGHT = 0.35
PROB_SOFTREPAIR_ADAPTIVE_OVERLAP_PENALTY = 0.10
PROB_SOFTREPAIR_SKIP_HEAVY_REPAIR_IF_NEAR_DOM = 0.98
PROB_SOFTREPAIR_ACCEPT_WORSE_RATIO = 1.05
PROB_SOFTREPAIR_STRICT_ACCEPT_ADD_K = 8
PROB_SOFTREPAIR_LOCAL_ONLY_MAX_UNDOM_RATIO = 0.08
PROB_SOFTREPAIR_BETA = 0.35
PROB_SOFTREPAIR_VERBOSE = False # print debug data
PROB_LOCAL_REPAIR_MAX_UNDOM = 32
PROB_LOCAL_REPAIR_MAX_CANDIDATES = 160
PROB_LOCAL_REPAIR_MAX_TIME = 8
ILP_POLISH_MAX_N = 400
ILP_POLISH_TIME = 5
ILP_POLISH_VERBOSE = False

# micro-ILP repair for uncovered holes (v23 local hole repair)
MICRO_ILP_ENABLE = False # deafult=True
MICRO_ILP_TIME_LIMIT = 3
MICRO_ILP_MAX_HOLE_SIZE = 64
MICRO_ILP_MAX_REGION_SIZE = 180
MICRO_ILP_CONFLICT_ENABLE = True
MICRO_ILP_EXPAND_HOPS = 2
MICRO_ILP_CONFLICT_SELECTED_HOPS = 2
MICRO_ILP_AFFECTED_HOPS = 1
ADAPTIVE_SUBGRAPH_REFINE_ENABLE = True
ADAPTIVE_SUBGRAPH_REFINE_MAX_REGION_SIZE = 220
ADAPTIVE_SUBGRAPH_REFINE_TIME_LIMIT = 5
ADAPTIVE_SUBGRAPH_REFINE_EXPAND_HOPS = 2
ADAPTIVE_SUBGRAPH_REFINE_AFFECTED_HOPS = 1
MICRO_ILP_OBJECTIVE_EPS = 1e-3

# GNN completion mode: "greedy", "beam", or "auto"
GNN_COMPLETION_MODE = "beam"
BEAM_ENABLE = True
BEAM_WIDTH = 8
BEAM_MAX_STEPS = 64
BEAM_CANDIDATE_TOPK = 20
BEAM_MAX_CANDIDATES_PER_STATE = 12

# v17 hybrid beam-search heuristics
BEAM_USE_HYBRID_SCORE = True
BEAM_SCORE_ALPHA = 1.0      # GNN probability weight
BEAM_SCORE_BETA = 1.5       # uncovered domination gain weight
BEAM_SCORE_GAMMA = 0.10     # overlap penalty weight
BEAM_SCORE_DELTA = 0.80     # RL policy score weight (v19)
BEAM_MIN_GAIN = 1           # prune weak expansions in beam (v19)
LOCAL_SWAP2_ENABLE = True   # enable 2-for-1 local improvement (v19)
LOCAL_SWAP2_MAX_TRIALS = 5000
BEAM_USE_SYMMETRY_PRUNING = True
BEAM_ADAPTIVE_WIDTH = False
BEAM_ADAPTIVE_WIDTH_FACTOR = 1.5

# =========================================================
#  Adaptive ILP teacher profiling
# =========================================================
ILP_TRAIN_MAX_NODES = 400   # hard cap; actual ILP usage is decided adaptively
ILP_ADAPTIVE_TEACHER = True
ILP_MIN_PROFILE_ATTEMPTS = 8
ILP_MIN_SUCCESS_RATE = 0.70
ILP_MAX_AVG_TIME = 15.0
ILP_PROFILE_BUCKETS = [
    (0, 150),
    (151, 200),
    (201, 300),
    (301, 400),
]

# =========================================================
#  Global Settings（追加）
# =========================================================

# GNN 的層數（所有模型共用）
GLOBAL_GNN_Layer = 4
#GLOBAL_GNN_Layer = 6
GPSCONV_GNN_LAYERS = 2
GPSCONV_HIDDEN_DIM = 32
GPSCONV_HEADS = 2
DEFAULT_HIDDEN_DIM = 64

CURRICULUM_SMALL_N = 150
CURRICULUM_MED_N = 400
CURRICULUM_LARGE_N = 900

RL_FINE_TUNE_EPOCHS = 10
RL_FINE_TUNE_LR = 1e-4

RL_FINE_EPOSODES = 200
RL_BETA = 0.3
RL_LAMBDA = 0.3
RL_ENTROPY = 0.03
RL_ENTROPY_MIN = 0.001
RL_ENTROPY_DECAY = 0.985
RL_BASELINE_MOMENTUM = 0.90
RL_REWARD_NORM_EPS = 1e-8
RL_USE_REWARD_NORMALIZATION = True
RL_ADVANTAGE_CLAMP = 5.0
USE_RL_FINE_TUNE = True
RL_VALIDATE_GRAPHS = True
RL_SAFE_SAVE_TO_CPU = True
RL_DEBUG_SYNC_CUDA = False
# RL safety / rollout defaults
RL_SKIP_INVALID_GRAPHS = True          # 遇到異常 graph 時自動略過，避免整個 RL 中斷
RL_MAX_GRAPH_N_FOR_RL = 700            # 全域上限；避免超大圖進入 RL
RL_MAX_GRAPH_N_FOR_RL_GPSCONV = 450    # GPSCONV 在 6GB 顯卡上更容易 OOM
RL_MAX_STEPS_PER_GRAPH = 64            # 單張圖 rollout 上限
RL_PER_GRAPH_UPDATE = True             # 每張圖立即 backward/step，避免累積整個 episode 的計算圖
RL_ACTION_MASK_DOMINATED = False       # True: 也遮罩已支配點；False: 只遮罩已選點
RL_REWARD_USE_INCREMENTAL_GAIN = True  # 使用每一步新增支配增益作為 reward 主體
RL_REWARD_GAIN_WEIGHT = 1.0            # 新增支配增益權重
RL_REWARD_REDUNDANCY_PENALTY = 0.1     # 冗餘選點懲罰
RL_TERMINAL_DOMINATION_BONUS = 1.25    # 全支配完成的終端獎勵
RL_UNCOVERED_PENALTY = 1.10            # 未完成支配時的終端懲罰
RL_STEP_SIZE_PENALTY = 0.02            # 每步選點成本
RL_REPEAT_ACTION_PENALTY = 0.05        # 重複/無效動作懲罰
RL_STATE_EXTRA_DIM = 5
RL_STATE_FEATURE_NAMES = (
    "is_selected",
    "is_dominated",
    "residual_undominated_neighbors",
    "marginal_gain",
    "step_index_over_N",
)

# v35: shrink raw set more aggressively while keeping raw domination high
RL_STOP_MIN_COVERAGE = 0.70
RL_STOP_TARGET_COVERAGE = 0.98
RL_STOP_TOO_EARLY_PENALTY = 1.00
RL_STOP_NEAR_DONE_BONUS = 0.15
RL_SHRINK_FUTURE_PENALTY_WEIGHT = 1.10
RL_SHRINK_DELTA_COVERAGE_WEIGHT = 0.20
RL_SHRINK_CURRENT_SIZE_WEIGHT = 0.25
RL_SHRINK_DONE_SIZE_WEIGHT = 0.35
RL_SHRINK_LOW_GAIN_PENALTY = 0.20
RL_FINAL_COMPLETED_SIZE_WEIGHT = 1.25
RL_FINAL_RAW_SIZE_WEIGHT = 0.45
RL_FINAL_RAW_DOM_WEIGHT = 1.00


# =========================================================
#  實驗模式：多尺寸 + 多挖洞比例（exp 版預設）
# =========================================================
EXPERIMENT_GRID_SIZES = [
    (20, 20),
    (30, 30),
    (40, 40),
    (60, 60),
    (70, 70),
]

EXPERIMENT_HOLE_RATES = [
    0.00,
    0.20,
    0.40,
    0.60,
    0.80,
]

# =========================================================
#  其餘 import / 定義
# =========================================================

import csv
from collections import defaultdict

# Tkinter GUI
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog

# =========================================================
#  鄰接規則 (4-鄰 / 8-鄰)
# =========================================================

NEIGHBOR_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBOR_8 = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
]

# =========================================================
#  隨機種子
# =========================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)

seed_everything(42)

# =========================================================
#  建圖與特徵
# =========================================================

def build_full_grid(m, n):
    """
    建立 m x n 的完整 grid or supergrid 的 adjacency (list[set]) 與座標
    """
    coords = []
    for i in range(m):
        for j in range(n):
            coords.append((i, j))
    coords = [(int(i), int(j)) for (i, j) in coords]

    if GRAPH_TOPOLOGY == "supergrid":
        neighbors = NEIGHBOR_8
    else:
        neighbors = NEIGHBOR_4

    adj = [set() for _ in range(m * n)]

    def idx(i, j):
        return i * n + j

    for i in range(m):
        for j in range(n):
            v = idx(i, j)
            for di, dj in neighbors:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n:
                    u = idx(ni, nj)
                    adj[v].add(u)
                    adj[u].add(v)

    return adj, coords

def build_irregular_grid_adj(m, n, hole_ratio=0.2, max_trials=50, ensure_connected=True):
    """
    建立「隨機挖洞」的非矩形格狀 / 超網格圖：
    - 先假設有一個 m x n 的完整格子
    - 隨機挖掉約 hole_ratio 比例的節點
    - 僅保留連通的最大元件（若 ensure_connected=True) 
      ==> 挖洞時檢查是否connected, 若否, 則不挖洞, 並往下一個挖洞點找; 若連續i次不成功, 則停止挖洞並記錄挖洞比例
    - 回傳：
        adj   : list[set]，長度 = N_irregular
        coords: list[(i, j)]，長度 = N_irregular, 對應每個 index 的格座標

    ✅ 新版 v2.-:
        - 若 GRAPH_TOPOLOGY == "grid"      → 邊只接 4-鄰居
        - 若 GRAPH_TOPOLOGY == "supergrid" → 邊接 8-鄰居（含對角線）
    """
    if GRAPH_TOPOLOGY == "supergrid":
        neighbors = NEIGHBOR_8
    else:
        neighbors = NEIGHBOR_4

    global GLOBAL_HOLE
    GLOBAL_HOLE = 0.0

    base_adj, base_coords = build_full_grid(m, n)
    N_full = len(base_adj)
    all_nodes = list(range(N_full))

    target_holes = int(hole_ratio * N_full)

    def is_connected_subgraph(adj_sub):
        if not adj_sub:
            return True
        visited = set()
        start = 0
        while start < len(adj_sub) and len(adj_sub[start]) == 0:
            start += 1
        if start == len(adj_sub):
            return False
        visited.add(start)
        stack = [start]
        while stack:
            v = stack.pop()
            for u in adj_sub[v]:
                if u not in visited:
                    visited.add(u)
                    stack.append(u)
        cnt = sum(1 for neighbors in adj_sub if neighbors)
        return (len(visited) == cnt)

    for trial in range(max_trials):
        random.shuffle(all_nodes)
        holes = set(all_nodes[:target_holes])
        keep  = [v for v in all_nodes if v not in holes]

        mapping = {v: i for i, v in enumerate(keep)}
        N_ir = len(keep)
        adj_ir = [set() for _ in range(N_ir)]
        coords_ir = [None] * N_ir

        for v in keep:
            i_new = mapping[v]
            coords_ir[i_new] = base_coords[v]
            for u in base_adj[v]:
                if u in keep:
                    j_new = mapping[u]
                    adj_ir[i_new].add(j_new)

        if not ensure_connected:
            actual_hole_ratio = 1.0 - (N_ir / N_full)
            GLOBAL_HOLE = actual_hole_ratio
            return adj_ir, coords_ir, GLOBAL_HOLE

        if is_connected_subgraph(adj_ir):
            actual_hole_ratio = 1.0 - (N_ir / N_full)
            GLOBAL_HOLE = actual_hole_ratio
            return adj_ir, coords_ir, GLOBAL_HOLE
        else:
            while target_holes > 0:
                target_holes -= 1
                holes = set(all_nodes[:target_holes])
                keep  = [v for v in all_nodes if v not in holes]
                mapping = {v: i for i, v in enumerate(keep)}
                N_ir = len(keep)
                adj_ir = [set() for _ in range(N_ir)]
                coords_ir = [None] * N_ir

                for v in keep:
                    i_new = mapping[v]
                    coords_ir[i_new] = base_coords[v]
                    for u in base_adj[v]:
                        if u in keep:
                            j_new = mapping[u]
                            adj_ir[i_new].add(j_new)

                if is_connected_subgraph(adj_ir):
                    actual_hole_ratio = 1.0 - (N_ir / N_full)
                    GLOBAL_HOLE = actual_hole_ratio
                    return adj_ir, coords_ir, GLOBAL_HOLE

    adj_ir = [set() for _ in range(N_full)]
    coords_ir = list(base_coords)
    GLOBAL_HOLE = 0.0
    return adj_ir, coords_ir, GLOBAL_HOLE

# =========================================================
#  各種特徵
# =========================================================

def compute_degree_features(adj):
    N = len(adj)
    deg = np.array([len(nei) for nei in adj], dtype=np.float32)
    max_deg = deg.max() if deg.max() > 0 else 1.0
    deg_norm = deg / max_deg
    return deg, deg_norm

def compute_shortest_path_eccentricity(adj):
    N = len(adj)
    INF = 10**9
    ecc = np.zeros(N, dtype=np.float32)
    for s in range(N):
        dist = np.full(N, INF, dtype=np.int32)
        dist[s] = 0
        queue = [s]
        head = 0
        while head < len(queue):
            v = queue[head]
            head += 1
            for u in adj[v]:
                if dist[u] > dist[v] + 1:
                    dist[u] = dist[v] + 1
                    queue.append(u)
        ecc[s] = max(d for d in dist if d < INF)
    max_ecc = ecc.max() if ecc.max() > 0 else 1.0
    ecc_norm = ecc / max_ecc
    return ecc_norm

def compute_laplacian_pe(adj, k=8):
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
    except Exception:
        # 🔹 沒有 scipy → 回傳隨機特徵（仍可訓練，只是沒 Laplacian 資訊）
        N = len(adj)
        return np.random.randn(N, k).astype(np.float32)

    N = len(adj)
    row, col, data = [], [], []
    for i in range(N):
        for j in adj[i]:
            row.append(i)
            col.append(j)
            data.append(1.0)

    A = sp.csr_matrix((data, (row, col)), shape=(N, N))
    deg = np.array(A.sum(axis=1)).reshape(-1)
    D = sp.diags(deg)
    L = D - A
    try:
        k_eff = min(k, N - 1)
        vals, vecs = spla.eigs(L, k=k_eff, which="SR")
        vecs = np.real(vecs)
    except Exception:
        vecs = np.random.randn(N, k).astype(np.float32)

    return vecs.astype(np.float32)

def compute_random_walk_embedding(adj, dim=16, walk_len=10, num_walks=5):
    N = len(adj)
    emb = np.zeros((N, dim), dtype=np.float32)
    rng = np.random.default_rng(123)

    for start in range(N):
        counts = np.zeros(N, dtype=np.float32)
        for _ in range(num_walks):
            cur = start
            for _w in range(walk_len):
                neighbors = list(adj[cur])
                if not neighbors:
                    break
                cur = rng.choice(neighbors)
                counts[cur] += 1.0
        total = counts.sum()
        if total > 0:
            counts /= total
        if N <= dim:
            emb[start, :N] = counts
        else:
            cum = np.cumsum(counts)
            for d in range(dim):
                thr = (d + 1) / (dim + 1)
                idx = np.searchsorted(cum, thr)
                if idx >= N:
                    idx = N - 1
                emb[start, d] = idx / float(N)
    maxv = np.max(np.abs(emb)) if np.max(np.abs(emb)) > 0 else 1.0
    emb /= maxv
    return emb

def compute_is_near_hole_feature(m, n, coords):
    """
    Boundary-awareness for irregular grids.
    A vertex is marked as near a hole if at least one in-bounds neighboring lattice
    position (4-neighbor for grid, 8-neighbor for supergrid) is missing from coords.
    Out-of-bounds positions are treated as outer boundary, not as holes.
    """
    if coords is None:
        return np.zeros((0,), dtype=np.float32)

    coord_list = [tuple(map(int, c)) for c in coords]
    coord_set = set(coord_list)
    neighbors = NEIGHBOR_8 if GRAPH_TOPOLOGY == "supergrid" else NEIGHBOR_4
    feat = np.zeros(len(coord_list), dtype=np.float32)

    for idx, (r, c) in enumerate(coord_list):
        near_hole = 0.0
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and (nr, nc) not in coord_set:
                near_hole = 1.0
                break
        feat[idx] = near_hole
    return feat


def get_dynamic_teacher_weights(n_nodes, ilp_weight=ILP_WEIGHT, greedy_weight=GREEDY_WEIGHT):
    """
    Small graphs use a much stronger ILP teacher so the model learns exact patterns.
    Larger graphs revert to the baseline dual-teacher mix.
    """
    if int(n_nodes) < int(SMALL_GRAPH_NODE_THRESHOLD):
        return float(SMALL_GRAPH_ILP_WEIGHT), float(SMALL_GRAPH_GREEDY_WEIGHT)
    return float(ilp_weight), float(greedy_weight)


def apply_symmetry_augmentation_to_data(
    data,
    enable=SYMMETRY_AUGMENTATION,
    swap_rc_prob=SYMMETRY_SWAP_RC_PROB,
    flip_row_prob=SYMMETRY_FLIP_ROW_PROB,
    flip_col_prob=SYMMETRY_FLIP_COL_PROB,
):
    """
    Feature-level symmetry augmentation.
    The first two channels are assumed to be row_norm / col_norm.
    We randomly flip row / col and optionally swap row<->col to simulate symmetric layouts
    without recomputing ILP labels.
    """
    if (not enable) or data is None or (not hasattr(data, "x")) or data.x is None or data.x.size(1) < 2:
        return data

    aug = copy.copy(data)
    aug.x = data.x.clone()

    if random.random() < float(flip_row_prob):
        aug.x[:, 0] = 1.0 - aug.x[:, 0]
    if random.random() < float(flip_col_prob):
        aug.x[:, 1] = 1.0 - aug.x[:, 1]
    if random.random() < float(swap_rc_prob):
        row_feat = aug.x[:, 0].clone()
        col_feat = aug.x[:, 1].clone()
        aug.x[:, 0] = col_feat
        aug.x[:, 1] = row_feat

    if hasattr(data, "edge_index"):
        aug.edge_index = data.edge_index
    if hasattr(data, "coords"):
        aug.coords = data.coords
    return aug


def build_full_features(m, n, adj, coords=None, pe_dim=8, rwe_dim=16):
    N = len(adj)
    if coords is None:
        coords = []
        for i in range(m):
            for j in range(n):
                coords.append((i, j))
    coords = np.array(coords, dtype=np.float32)

    deg, deg_norm = compute_degree_features(adj)
    ecc_norm = compute_shortest_path_eccentricity(adj)
    lap_pe = compute_laplacian_pe(adj, k=pe_dim)
    rwe = compute_random_walk_embedding(adj, dim=rwe_dim)
    is_near_hole = compute_is_near_hole_feature(m, n, coords)

    max_r = max(c[0] for c in coords) if len(coords) > 0 else 1
    max_c = max(c[1] for c in coords) if len(coords) > 0 else 1
    max_r = max(max_r, 1)
    max_c = max(max_c, 1)
    row_norm = coords[:, 0] / max_r
    col_norm = coords[:, 1] / max_c

    feat_list = []
    feat_list.append(row_norm.reshape(-1, 1))
    feat_list.append(col_norm.reshape(-1, 1))
    feat_list.append(is_near_hole.reshape(-1, 1))
    feat_list.append(deg_norm.reshape(-1, 1))
    feat_list.append(ecc_norm.reshape(-1, 1))
    feat_list.append(lap_pe)
    feat_list.append(rwe)

    x = np.concatenate(feat_list, axis=1).astype(np.float32)

    # Use bidirectional edges so the GNN sees the same undirected topology as `adj`.
    edges = []
    for i in range(N):
        for j in adj[i]:
            edges.append((i, j))
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(
        x=torch.tensor(x, dtype=DTYPE),
        edge_index=edge_index,
    )
    data.coords = torch.tensor(coords, dtype=DTYPE)
    return data


# =========================================================
#  ILP
# =========================================================

ILP_PROFILE_STATS = {}

def _init_ilp_profile_stats():
    global ILP_PROFILE_STATS
    ILP_PROFILE_STATS = {
        bucket: {"attempts": 0, "success": 0, "times": [], "skipped": 0}
        for bucket in ILP_PROFILE_BUCKETS
    }


def reset_ilp_profile_stats():
    _init_ilp_profile_stats()


def _find_ilp_bucket(n_nodes):
    for lo, hi in ILP_PROFILE_BUCKETS:
        if lo <= n_nodes <= hi:
            return (lo, hi)
    return None


def _avg(values):
    return sum(values) / len(values) if values else 0.0


def get_ilp_profile_snapshot():
    rows = []
    for bucket in ILP_PROFILE_BUCKETS:
        st = ILP_PROFILE_STATS.get(bucket, {"attempts": 0, "success": 0, "times": [], "skipped": 0})
        attempts = int(st.get("attempts", 0))
        success = int(st.get("success", 0))
        skipped = int(st.get("skipped", 0))
        success_rate = (success / attempts) if attempts > 0 else 0.0
        avg_time = _avg(st.get("times", []))
        rows.append({
            "bucket": bucket,
            "attempts": attempts,
            "success": success,
            "success_rate": success_rate,
            "avg_time": avg_time,
            "skipped": skipped,
        })
    return rows


def print_ilp_profile_summary():
    print("[ILP Profile] bucket summary:")
    for row in get_ilp_profile_snapshot():
        lo, hi = row["bucket"]
        print(
            f"  nodes[{lo:>3},{hi:>3}] | attempts={row['attempts']:>4} "
            f"success={row['success']:>4} rate={row['success_rate']:.2%} "
            f"avg_time={row['avg_time']:.2f}s skipped={row['skipped']:>4}"
        )


def should_attempt_ilp_teacher(n_nodes):
    if n_nodes > ILP_TRAIN_MAX_NODES:
        return False, f"N_nodes={n_nodes} > hard cap {ILP_TRAIN_MAX_NODES}"

    if not ILP_ADAPTIVE_TEACHER:
        return True, "adaptive teacher disabled"

    bucket = _find_ilp_bucket(n_nodes)
    if bucket is None:
        return False, f"N_nodes={n_nodes} outside profiling buckets"

    st = ILP_PROFILE_STATS.get(bucket)
    if st is None:
        return True, f"bucket {bucket} has no stats yet"

    attempts = int(st.get("attempts", 0))
    if attempts < ILP_MIN_PROFILE_ATTEMPTS:
        return True, f"warm-up profiling for bucket {bucket} ({attempts}/{ILP_MIN_PROFILE_ATTEMPTS})"

    success_rate = st.get("success", 0) / max(attempts, 1)
    avg_time = _avg(st.get("times", []))

    if success_rate < ILP_MIN_SUCCESS_RATE:
        return False, f"bucket {bucket} success_rate={success_rate:.2%} < {ILP_MIN_SUCCESS_RATE:.0%}"
    if avg_time > ILP_MAX_AVG_TIME:
        return False, f"bucket {bucket} avg_time={avg_time:.2f}s > {ILP_MAX_AVG_TIME:.2f}s"

    return True, f"bucket {bucket} ok (success_rate={success_rate:.2%}, avg_time={avg_time:.2f}s)"


def _record_ilp_attempt(n_nodes, success, elapsed):
    bucket = _find_ilp_bucket(n_nodes)
    if bucket is None:
        return
    st = ILP_PROFILE_STATS.setdefault(bucket, {"attempts": 0, "success": 0, "times": [], "skipped": 0})
    st["attempts"] += 1
    if success:
        st["success"] += 1
    st["times"].append(float(elapsed))


def _record_ilp_skip(n_nodes):
    bucket = _find_ilp_bucket(n_nodes)
    if bucket is None:
        return
    st = ILP_PROFILE_STATS.setdefault(bucket, {"attempts": 0, "success": 0, "times": [], "skipped": 0})
    st["skipped"] += 1


def _normalize_ilp_status(status: str) -> str:
    s = str(status).strip().lower()
    if s in ("optimal", "empty"):
        return "OPTIMAL"
    if s in ("integer feasible", "integerfeasible", "feasible"):
        return "TIME_LIMIT_FEASIBLE"
    if s in ("not solved", "undefined", "infeasible", "unbounded"):
        return "FAILED"
    if s in ("pulp_missing",):
        return "PULP_MISSING"
    if s in ("skipped",):
        return "SKIPPED"
    return str(status)


def ilp_minimum_dominating_set_with_info(adj, time_limit=ILP_MaxTime):
    N = len(adj)
    if N == 0:
        return [], {
            "attempted": True,
            "success": True,
            "status": "Empty",
            "status_class": "OPTIMAL",
            "optimal": True,
            "time_sec": 0.0,
            "fallback": False,
        }

    t0 = time.perf_counter()
    try:
        import pulp
    except ImportError:
        elapsed = time.perf_counter() - t0
        print("[WARN] pulp not installed, fallback to greedy dominating set.")
        sol = greedy_dominating_set(adj)
        return sol, {
            "attempted": False,
            "success": False,
            "status": "PULP_MISSING",
            "status_class": "PULP_MISSING",
            "optimal": False,
            "time_sec": elapsed,
            "fallback": True,
        }

    prob = pulp.LpProblem("MinDomSet", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat=pulp.LpBinary) for i in range(N)]

    prob += pulp.lpSum(x[i] for i in range(N))

    for v in range(N):
        neighbors = list(adj[v]) + [v]
        prob += pulp.lpSum(x[u] for u in neighbors) >= 1

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    prob.solve(solver)

    elapsed = time.perf_counter() - t0
    status = pulp.LpStatus[prob.status]
    status_class = _normalize_ilp_status(status)
    optimal = (status_class == "OPTIMAL")
    success = (status_class in ("OPTIMAL", "TIME_LIMIT_FEASIBLE"))

    if not success:
        print(f"[ILP] Status={status}, fallback to greedy.")
        sol = greedy_dominating_set(adj)
        return sol, {
            "attempted": True,
            "success": False,
            "status": status,
            "status_class": status_class,
            "optimal": False,
            "time_sec": elapsed,
            "fallback": True,
        }

    sol = []
    for i in range(N):
        if x[i].varValue is not None and x[i].varValue > 0.5:
            sol.append(i)
    return sol, {
        "attempted": True,
        "success": True,
        "status": status,
        "status_class": status_class,
        "optimal": optimal,
        "time_sec": elapsed,
        "fallback": False,
    }


def choose_teacher_solutions_for_training(adj, n_nodes):
    sol_gr = greedy_dominating_set(adj)
    use_ilp, reason = should_attempt_ilp_teacher(n_nodes)

    if not use_ilp:
        _record_ilp_skip(n_nodes)
        return sol_gr, sol_gr, {
            "teacher_mode": "GreedyOnly",
            "reason": reason,
            "ilp_attempted": False,
            "ilp_success": False,
            "ilp_time_sec": 0.0,
            "ilp_status": "SKIPPED",
            "ilp_status_class": "SKIPPED",
            "ilp_optimal": False,
        }

    sol_ilp, info = solve_domination_ilp(adj, timeout=DATASET_ILP_TIMEOUT, return_info=True)
    _record_ilp_attempt(n_nodes, info.get("success", False), info.get("time_sec", 0.0))
    teacher_mode = "DualTeacher" if info.get("success", False) else "GreedyFallback"

    return sol_ilp, sol_gr, {
        "teacher_mode": teacher_mode,
        "reason": reason,
        "ilp_attempted": bool(info.get("attempted", False)),
        "ilp_success": bool(info.get("success", False)),
        "ilp_time_sec": float(info.get("time_sec", 0.0)),
        "ilp_status": str(info.get("status", "UNKNOWN")),
        "ilp_status_class": str(info.get("status_class", _normalize_ilp_status(info.get("status", "UNKNOWN")))),
        "ilp_optimal": bool(info.get("optimal", False)),
    }


#reset_ilp_profile_stats()


def _format_ilp_info_short(ilp_info: dict) -> str:
    """Compact human-readable ILP status for menu 12 / plotting comparisons."""
    if not isinstance(ilp_info, dict):
        return "ILP info unavailable"
    status = str(ilp_info.get("status_class", ilp_info.get("status", "UNKNOWN")))
    optimal = bool(ilp_info.get("optimal", False))
    attempted = bool(ilp_info.get("attempted", False))
    fallback = bool(ilp_info.get("fallback", False))
    t = float(ilp_info.get("time_sec", 0.0) or 0.0)
    parts = [f"status={status}"]
    parts.append(f"optimal={'Y' if optimal else 'N'}")
    parts.append(f"attempted={'Y' if attempted else 'N'}")
    parts.append(f"fallback={'Y' if fallback else 'N'}")
    parts.append(f"time={t:.3f}s")
    return ", ".join(parts)


def ilp_minimum_dominating_set(adj, time_limit=ILP_MaxTime):
    sol, _info = ilp_minimum_dominating_set_with_info(adj, time_limit=time_limit)
    return sol


def solve_domination_ilp(adj, timeout=DATASET_ILP_TIMEOUT, return_info=False):
    """Compatibility wrapper for dataset-label generation with timeout protection."""
    sol, info = ilp_minimum_dominating_set_with_info(adj, time_limit=timeout)
    if return_info:
        return sol, info
    return sol

def greedy_dominating_set(adj):
    N = len(adj)
    D = set()
    undom = set(range(N))
    while undom:
        best_v = None
        best_cover = -1
        for v in range(N):
            cover = 0
            if v in undom:
                cover += 1
            for u in adj[v]:
                if u in undom:
                    cover += 1
            if cover > best_cover:
                best_cover = cover
                best_v = v
        newly_covered = {best_v}
        newly_covered |= set(adj[best_v])
        undom -= newly_covered
        D.add(best_v)
    return sorted(D)

def check_domination(adj, D):
    N = len(adj)
    Dset = set(D)
    for v in range(N):
        if v in Dset:
            continue
        ok = False
        if v in Dset:
            ok = True
        else:
            for u in adj[v]:
                if u in Dset:
                    ok = True
                    break
        if not ok:
            return False
    return True

def domination_coverage(adj, D):
    N = len(adj)
    Dset = set(D)
    dominated = set()
    for v in range(N):
        if v in Dset:
            dominated.add(v)
            for u in adj[v]:
                dominated.add(u)
    return len(dominated) / max(N, 1)

def compute_coverage(adj, D):
    """
    舊版本名稱相容：
    compute_coverage(adj, D) = domination_coverage(adj, D)
    """
    return domination_coverage(adj, D)

def greedy_completion_with_initial(adj, initial_set):
    """
    Greedy 補點版本：
    - 先把 initial_set 當作已選節點
    - 把被它們支配到的點標記為已覆蓋
    - 再用 greedy 繼續補，直到全圖被支配
    """
    N = len(adj)
    D = set(initial_set)

    undom = set(range(N))
    for v in list(D):
        if v in undom:
            undom.remove(v)
        for u in adj[v]:
            if u in undom:
                undom.remove(u)

    while undom:
        best_v = None
        best_cover = -1
        for v in range(N):
            cover = 0
            if v in undom:
                cover += 1
            for u in adj[v]:
                if u in undom:
                    cover += 1
            if cover > best_cover:
                best_cover = cover
                best_v = v
        newly_covered = {best_v}
        newly_covered |= set(adj[best_v])
        undom -= newly_covered
        D.add(best_v)

    return sorted(D)

def _probs_to_numpy_scores(probs):
    import numpy as np
    if isinstance(probs, torch.Tensor):
        scores = probs.detach().cpu().numpy().reshape(-1)
    else:
        scores = np.array(probs, dtype=np.float32).reshape(-1)
    return np.clip(scores.astype(np.float32), 0.0, 1.0)


def _dominated_vertices_set(adj, D):
    Dset = set(D)
    dominated = set()
    for v in Dset:
        dominated.add(v)
        dominated.update(adj[v])
    return dominated


def _select_raw_by_fixed_threshold(adj, scores, threshold):
    N = len(adj)
    S_raw = {i for i in range(N) if scores[i] >= threshold}
    if not S_raw and N > 0:
        S_raw.add(int(scores.argmax()))
    return sorted(S_raw)


def _adaptive_threshold_raw_selection(
    adj,
    scores,
    base_threshold=PROB_SOFTREPAIR_THRESHOLD,
    target_coverage=PROB_SOFTREPAIR_TARGET_RAW_COVERAGE,
    min_ratio=PROB_SOFTREPAIR_MIN_RAW_RATIO,
    max_ratio=PROB_SOFTREPAIR_MAX_RAW_RATIO,
    gain_weight=PROB_SOFTREPAIR_ADAPTIVE_GAIN_WEIGHT,
    score_weight=PROB_SOFTREPAIR_ADAPTIVE_SCORE_WEIGHT,
    overlap_penalty=PROB_SOFTREPAIR_ADAPTIVE_OVERLAP_PENALTY,
):
    """
    Coverage-aware adaptive thresholding.

    Start from the fixed-threshold seed and then greedily add high-value
    vertices until a target raw coverage ratio is reached or a size cap is hit.
    This avoids the severe under-selection often caused by one global threshold.
    """
    N = len(adj)
    if N == 0:
        return []

    min_keep = max(1, int(math.ceil(min_ratio * N)))
    max_keep = max(min_keep, int(math.ceil(max_ratio * N)))

    selected = set(_select_raw_by_fixed_threshold(adj, scores, base_threshold))
    score_order = list(np.argsort(-scores))
    if len(selected) < min_keep:
        for v in score_order:
            selected.add(int(v))
            if len(selected) >= min_keep:
                break

    dominated = _dominated_vertices_set(adj, selected)

    def coverage_ratio():
        return len(dominated) / max(N, 1)

    while len(selected) < max_keep and coverage_ratio() < target_coverage:
        best_v = None
        best_key = None
        for v in range(N):
            if v in selected:
                continue
            closed = get_closed_neighborhood(adj, v)
            gain = len(closed - dominated)
            overlap = len(closed & dominated)
            key = (
                gain_weight * float(gain)
                + score_weight * float(scores[v])
                - overlap_penalty * float(overlap) / max(len(closed), 1),
                gain,
                float(scores[v]),
                -overlap,
                -v,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_v = int(v)
        if best_v is None:
            break
        selected.add(best_v)
        dominated.update(get_closed_neighborhood(adj, best_v))

    return sorted(selected)


def gnn_raw_only(
    adj,
    probs,
    threshold=PROB_SOFTREPAIR_THRESHOLD,
    use_adaptive_threshold=PROB_SOFTREPAIR_USE_ADAPTIVE_THRESHOLD,
    target_coverage=PROB_SOFTREPAIR_TARGET_RAW_COVERAGE,
):
    scores = _probs_to_numpy_scores(probs)
    N = len(adj)
    if len(scores) != N:
        raise ValueError(f"[gnn_raw_only] len(scores)={len(scores)} != N={N}")
    if use_adaptive_threshold:
        return _adaptive_threshold_raw_selection(
            adj,
            scores,
            base_threshold=threshold,
            target_coverage=target_coverage,
        )
    return _select_raw_by_fixed_threshold(adj, scores, threshold)
def get_undominated_vertices(adj, D):
    Dset = set(D)
    undom = []
    for v in range(len(adj)):
        if v in Dset:
            continue
        dominated = False
        for u in adj[v]:
            if u in Dset:
                dominated = True
                break
        if not dominated:
            undom.append(v)
    return undom


def prune_redundant_vertices(adj, D, probs=None, verbose=False):
    """
    Iteratively remove redundant vertices while preserving domination.
    If probs are given, lower-probability vertices are tried first so that
    higher-confidence GNN vertices are kept whenever possible.
    """
    scores = _probs_to_numpy_scores(probs) if probs is not None else None
    Dset = set(D)
    if not Dset:
        return []

    changed = True
    rounds = 0
    removed_total = 0
    while changed:
        changed = False
        rounds += 1
        order = list(Dset)
        if scores is not None:
            order.sort(key=lambda v: (scores[v], len(adj[v]), v))
        else:
            order.sort(key=lambda v: (len(adj[v]), v))

        for v in order:
            if v not in Dset:
                continue
            Dset.remove(v)
            if check_domination(adj, Dset):
                changed = True
                removed_total += 1
            else:
                Dset.add(v)

    if verbose:
        print(f"[Prune] rounds={rounds}, removed={removed_total}, final={len(Dset)}")
    return sorted(Dset)


def guided_greedy_completion_with_initial(adj, initial_set, probs=None, verbose=False):
    """
    Global greedy completion initialized by initial_set, with optional GNN-score
    guidance as a tie-breaker. This is intentionally global (not local ILP), so
    it can pick a far-away but high-coverage vertex when beneficial.
    """
    scores = _probs_to_numpy_scores(probs) if probs is not None else None
    N = len(adj)
    D = set(initial_set)

    undom = set(range(N))
    for v in list(D):
        undom.discard(v)
        for u in adj[v]:
            undom.discard(u)

    steps = 0
    while undom:
        best_v = None
        best_key = None
        for v in range(N):
            cover = (1 if v in undom else 0)
            for u in adj[v]:
                if u in undom:
                    cover += 1
            score_v = float(scores[v]) if scores is not None else 0.0
            # Prefer larger uncovered coverage first, then higher GNN probability,
            # then larger degree. Final tie-break by smaller index for determinism.
            key = (cover, score_v, len(adj[v]), -v)
            if best_key is None or key > best_key:
                best_key = key
                best_v = v

        newly_covered = {best_v} | set(adj[best_v])
        undom -= newly_covered
        D.add(best_v)
        steps += 1

    if verbose:
        print(f"[GuidedGreedy] added={len(D) - len(set(initial_set))}, final={len(D)}, steps={steps}")
    return sorted(D)



def _compute_undominated_set(adj, Dset):
    undom = set(range(len(adj)))
    for v in Dset:
        undom.discard(v)
        for u in adj[v]:
            undom.discard(u)
    return undom


def _beam_cover_stats(adj, Dset, undom, v):
    closed = get_closed_neighborhood(adj, v)
    gain = len(closed & undom)
    overlap = len(closed) - gain
    return gain, overlap


def _beam_candidate_priority(adj, Dset, undom, v, scores, rl_scores=None):
    gain, overlap = _beam_cover_stats(adj, Dset, undom, v)
    score_v = float(scores[v]) if scores is not None else 0.0
    rl_v = float(rl_scores[v]) if rl_scores is not None else 0.0
    if BEAM_USE_HYBRID_SCORE:
        hybrid = (
            BEAM_SCORE_ALPHA * score_v
            + BEAM_SCORE_BETA * gain
            - BEAM_SCORE_GAMMA * overlap
            + BEAM_SCORE_DELTA * rl_v
        )
    else:
        hybrid = gain
    return (hybrid, gain, score_v, rl_v, -overlap, len(adj[v]), -v)


def _beam_state_sort_key(item):
    Dset, undom, state_score, avg_prob = item
    return (len(undom), len(Dset), -state_score, -avg_prob, tuple(sorted(Dset)))


def beam_search_completion_with_initial(
    adj,
    initial_set,
    probs=None,
    rl_probs=None,
    beam_width=BEAM_WIDTH,
    max_steps=BEAM_MAX_STEPS,
    candidate_topk=BEAM_CANDIDATE_TOPK,
    max_candidates_per_state=BEAM_MAX_CANDIDATES_PER_STATE,
    verbose=False,
):
    """
    v19 hybrid beam-search completion initialized from ``initial_set``.

    New features
    ------------
    1. Hybrid score = alpha * GNN + beta * uncovered_gain - gamma * overlap + delta * RL.
    2. Symmetry pruning via visited chosen-set hashes.
    3. Optional adaptive beam width.
    4. Minimum-gain pruning for weak branches.
    5. Deterministic tie-breaking for reproducible experiments.
    """
    scores = _probs_to_numpy_scores(probs) if probs is not None else None
    rl_scores = _probs_to_numpy_scores(rl_probs) if rl_probs is not None else None
    N = len(adj)
    D0 = set(initial_set)
    undom0 = _compute_undominated_set(adj, D0)

    if not undom0:
        if verbose:
            print(f"[Beam-v19] initial set already dominating, size={len(D0)}")
        return sorted(D0)

    eff_beam_width = int(beam_width)
    if BEAM_ADAPTIVE_WIDTH:
        scarcity = max(1.0, len(undom0) / max(1, N))
        eff_beam_width = max(1, int(round(beam_width * (1.0 + (BEAM_ADAPTIVE_WIDTH_FACTOR - 1.0) * scarcity))))

    def rank_candidates(Dset, undom):
        cand = []
        for v in range(N):
            if v in Dset:
                continue
            gain, overlap = _beam_cover_stats(adj, Dset, undom, v)
            if gain < BEAM_MIN_GAIN:
                continue
            priority = _beam_candidate_priority(adj, Dset, undom, v, scores, rl_scores=rl_scores)
            cand.append((priority, v))
        cand.sort(reverse=True)
        ranked = [v for _, v in cand[:candidate_topk]]
        if max_candidates_per_state and len(ranked) > max_candidates_per_state:
            ranked = ranked[:max_candidates_per_state]
        return ranked

    def state_metrics(Dset, undom):
        avg_prob = 0.0 if scores is None or not Dset else float(sum(scores[v] for v in Dset)) / max(1, len(Dset))
        state_score = 0.0
        for v in Dset:
            gain0 = len(get_closed_neighborhood(adj, v) & undom0)
            overlap0 = len(get_closed_neighborhood(adj, v)) - gain0
            score_v = float(scores[v]) if scores is not None else 0.0
            rl_v = float(rl_scores[v]) if rl_scores is not None else 0.0
            if BEAM_USE_HYBRID_SCORE:
                state_score += (
                    BEAM_SCORE_ALPHA * score_v
                    + BEAM_SCORE_BETA * gain0
                    - BEAM_SCORE_GAMMA * overlap0
                    + BEAM_SCORE_DELTA * rl_v
                )
            else:
                state_score += gain0
        return state_score, avg_prob

    def quality_tuple(Dset, undom, state_score, avg_prob):
        return (len(undom), len(Dset), -state_score, -avg_prob, tuple(sorted(Dset)))

    state_score0, avg_prob0 = state_metrics(D0, undom0)
    beam = [(set(D0), set(undom0), state_score0, avg_prob0)]
    best_state = beam[0]
    best_quality = quality_tuple(*best_state)
    visited = set()
    if BEAM_USE_SYMMETRY_PRUNING:
        visited.add(tuple(sorted(D0)))

    dynamic_step_limit = min(max_steps, max(1, len(undom0)))

    for step in range(dynamic_step_limit):
        expanded = []
        for Dset, undom, _, _ in beam:
            if not undom:
                if verbose:
                    print(f"[Beam-v19] solved at step={step}, size={len(Dset)}")
                return sorted(Dset)

            for v in rank_candidates(Dset, undom):
                newD = set(Dset)
                newD.add(v)
                hash_key = tuple(sorted(newD))
                if BEAM_USE_SYMMETRY_PRUNING and hash_key in visited:
                    continue
                newUndom = set(undom)
                newUndom -= get_closed_neighborhood(adj, v)
                new_state_score, new_avg_prob = state_metrics(newD, newUndom)
                expanded.append((newD, newUndom, new_state_score, new_avg_prob))
                if BEAM_USE_SYMMETRY_PRUNING:
                    visited.add(hash_key)

        if not expanded:
            break

        uniq = {}
        for item in expanded:
            Dset, undom, state_score, avg_prob = item
            key = tuple(sorted(Dset))
            cur = uniq.get(key)
            if cur is None or quality_tuple(Dset, undom, state_score, avg_prob) < quality_tuple(*cur):
                uniq[key] = item
        expanded = list(uniq.values())
        expanded.sort(key=_beam_state_sort_key)
        beam = expanded[:eff_beam_width]

        if beam:
            candidate_quality = quality_tuple(*beam[0])
            if candidate_quality < best_quality:
                best_state = beam[0]
                best_quality = candidate_quality

        if verbose:
            print(
                f"[Beam-v19] step={step+1}, frontier={len(expanded)}, keep={len(beam)}, "
                f"best_size={len(best_state[0])}, best_undom={len(best_state[1])}, width={eff_beam_width}"
            )

        if beam and not beam[0][1]:
            return sorted(beam[0][0])

    if verbose:
        print(
            f"[Beam-v19] incomplete after max_steps={dynamic_step_limit}; "
            f"best_size={len(best_state[0])}, best_undom={len(best_state[1])}"
        )
    return sorted(best_state[0])


def _resolve_completion_mode(mode=None):
    mode = (mode or GNN_COMPLETION_MODE or 'beam').strip().lower()
    if mode not in {'greedy', 'beam', 'auto'}:
        mode = 'beam'
    if mode == 'auto':
        return 'beam' if BEAM_ENABLE else 'greedy'
    if mode == 'beam' and not BEAM_ENABLE:
        return 'greedy'
    return mode


def get_completion_method_suffix(mode=None):
    mode = _resolve_completion_mode(mode)
    return 'PruneILPBeamGreedy'


def menu_set_gnn_completion_mode():
    global GNN_COMPLETION_MODE
    print("\n=== 設定 GNN 補點模式 ===")
    print(f"目前模式：{_resolve_completion_mode(GNN_COMPLETION_MODE)}")
    print('1. Beam Search 補點 (預設)')
    print('2. Guided Greedy 補點')
    print('3. Auto (若 Beam 啟用則用 Beam，否則用 Greedy)')
    choice = input('請選擇：').strip()
    if choice == '1':
        GNN_COMPLETION_MODE = 'beam'
    elif choice == '2':
        GNN_COMPLETION_MODE = 'greedy'
    elif choice == '3':
        GNN_COMPLETION_MODE = 'auto'
    else:
        print('❌ 無效選擇，維持原設定。')
        return
    print(f"✔ 已設定 GNN 補點模式為：{_resolve_completion_mode(GNN_COMPLETION_MODE)}")


def menu_set_beam_params():
    global BEAM_ENABLE, BEAM_WIDTH, BEAM_MAX_STEPS, BEAM_CANDIDATE_TOPK, BEAM_MAX_CANDIDATES_PER_STATE
    global BEAM_USE_HYBRID_SCORE, BEAM_SCORE_ALPHA, BEAM_SCORE_BETA, BEAM_SCORE_GAMMA, BEAM_SCORE_DELTA, BEAM_MIN_GAIN
    global BEAM_USE_SYMMETRY_PRUNING, BEAM_ADAPTIVE_WIDTH, BEAM_ADAPTIVE_WIDTH_FACTOR
    print("\n=== 設定 Beam Search 參數 ===")
    print(
        f"目前參數：enable={BEAM_ENABLE}, width={BEAM_WIDTH}, max_steps={BEAM_MAX_STEPS}, "
        f"topk={BEAM_CANDIDATE_TOPK}, max_per_state={BEAM_MAX_CANDIDATES_PER_STATE}, "
        f"hybrid={BEAM_USE_HYBRID_SCORE}, alpha={BEAM_SCORE_ALPHA}, beta={BEAM_SCORE_BETA}, gamma={BEAM_SCORE_GAMMA}, delta={BEAM_SCORE_DELTA}, min_gain={BEAM_MIN_GAIN}, "
        f"symmetry={BEAM_USE_SYMMETRY_PRUNING}, adaptive_width={BEAM_ADAPTIVE_WIDTH}, factor={BEAM_ADAPTIVE_WIDTH_FACTOR}"
    )
    enable_choice = input(f"啟用 Beam Search？ (y/n，直接 Enter 保持 {'y' if BEAM_ENABLE else 'n'})：").strip().lower()
    if enable_choice in {'y', 'yes', '1'}:
        BEAM_ENABLE = True
    elif enable_choice in {'n', 'no', '0'}:
        BEAM_ENABLE = False

    hybrid_choice = input(f"啟用 Hybrid Beam Score？ (y/n，直接 Enter 保持 {'y' if BEAM_USE_HYBRID_SCORE else 'n'})：").strip().lower()
    if hybrid_choice in {'y', 'yes', '1'}:
        BEAM_USE_HYBRID_SCORE = True
    elif hybrid_choice in {'n', 'no', '0'}:
        BEAM_USE_HYBRID_SCORE = False

    symmetry_choice = input(f"啟用 Symmetry Pruning？ (y/n，直接 Enter 保持 {'y' if BEAM_USE_SYMMETRY_PRUNING else 'n'})：").strip().lower()
    if symmetry_choice in {'y', 'yes', '1'}:
        BEAM_USE_SYMMETRY_PRUNING = True
    elif symmetry_choice in {'n', 'no', '0'}:
        BEAM_USE_SYMMETRY_PRUNING = False

    adaptive_choice = input(f"啟用 Adaptive Beam Width？ (y/n，直接 Enter 保持 {'y' if BEAM_ADAPTIVE_WIDTH else 'n'})：").strip().lower()
    if adaptive_choice in {'y', 'yes', '1'}:
        BEAM_ADAPTIVE_WIDTH = True
    elif adaptive_choice in {'n', 'no', '0'}:
        BEAM_ADAPTIVE_WIDTH = False

    BEAM_WIDTH = input_int_with_default("Beam width", BEAM_WIDTH, min_value=1)
    BEAM_MAX_STEPS = input_int_with_default("Beam max steps", BEAM_MAX_STEPS, min_value=1)
    BEAM_CANDIDATE_TOPK = input_int_with_default("Beam candidate topK", BEAM_CANDIDATE_TOPK, min_value=1)
    BEAM_MAX_CANDIDATES_PER_STATE = input_int_with_default("Beam max candidates per state", BEAM_MAX_CANDIDATES_PER_STATE, min_value=1)
    BEAM_SCORE_ALPHA = input_float_with_default("Beam alpha (GNN 權重)", BEAM_SCORE_ALPHA, min_value=0.0)
    BEAM_SCORE_BETA = input_float_with_default("Beam beta (gain 權重)", BEAM_SCORE_BETA, min_value=0.0)
    BEAM_SCORE_GAMMA = input_float_with_default("Beam gamma (overlap penalty)", BEAM_SCORE_GAMMA, min_value=0.0)
    BEAM_SCORE_DELTA = input_float_with_default("Beam delta (RL 權重)", BEAM_SCORE_DELTA, min_value=0.0)
    BEAM_MIN_GAIN = input_int_with_default("Beam minimum gain", BEAM_MIN_GAIN, min_value=0)
    BEAM_ADAPTIVE_WIDTH_FACTOR = input_float_with_default("Adaptive width factor", BEAM_ADAPTIVE_WIDTH_FACTOR, min_value=1.0)

    print(
        f"✔ Beam Search 參數已更新：enable={BEAM_ENABLE}, width={BEAM_WIDTH}, max_steps={BEAM_MAX_STEPS}, "
        f"topk={BEAM_CANDIDATE_TOPK}, max_per_state={BEAM_MAX_CANDIDATES_PER_STATE}, hybrid={BEAM_USE_HYBRID_SCORE}, "
        f"alpha={BEAM_SCORE_ALPHA}, beta={BEAM_SCORE_BETA}, gamma={BEAM_SCORE_GAMMA}, delta={BEAM_SCORE_DELTA}, min_gain={BEAM_MIN_GAIN}, "
        f"symmetry={BEAM_USE_SYMMETRY_PRUNING}, adaptive_width={BEAM_ADAPTIVE_WIDTH}, factor={BEAM_ADAPTIVE_WIDTH_FACTOR}"
    )


def dominating_set_loss(adj, D):
    """
    Smaller is better.
    Strongly penalize undominated vertices so any invalid set is much worse
    than a valid dominating set of similar size.
    """
    Dset = set(D)
    undom = 0
    for v in range(len(adj)):
        if v in Dset:
            continue
        ok = False
        for u in adj[v]:
            if u in Dset:
                ok = True
                break
        if not ok:
            undom += 1
    return len(Dset) + 1000 * undom


def get_closed_neighborhood(adj, v):
    return {v} | set(adj[v])


def local_swap_optimize_dominating_set(
    adj,
    D,
    probs=None,
    max_rounds=3,
    verbose=False,
):
    """
    Local improvement without ILP.

    Strategy:
    1) Remove redundant vertices whenever possible.
    2) Try 1-for-1 swaps u -> w and accept improving moves.
    3) Prune again after an accepted move.
    """
    import numpy as np

    scores = _probs_to_numpy_scores(probs) if probs is not None else np.zeros(len(adj), dtype=np.float32)
    Dset = set(D)

    if not check_domination(adj, Dset):
        if verbose:
            print("[LocalSwap] input set is not a dominating set; skip local swap.")
        return sorted(Dset)

    Dset = set(prune_redundant_vertices(adj, Dset, probs=scores, verbose=False))

    def current_loss(S):
        return dominating_set_loss(adj, S)

    improved_any = False

    for rd in range(max_rounds):
        improved = False

        changed = True
        while changed:
            changed = False
            selected_order = sorted(Dset, key=lambda x: (scores[x], x))
            for u in selected_order:
                T = set(Dset)
                T.remove(u)
                if check_domination(adj, T):
                    Dset = T
                    changed = True
                    improved = True
                    improved_any = True
                    if verbose:
                        print(f"[LocalSwap][round {rd+1}] remove redundant vertex {u}")
                    break

        base_loss = current_loss(Dset)
        best_move = None
        best_loss = base_loss

        selected_vertices = sorted(Dset, key=lambda x: (scores[x], x))
        candidate_add = set()
        for u in Dset:
            candidate_add |= get_closed_neighborhood(adj, u)
            for x in adj[u]:
                candidate_add |= get_closed_neighborhood(adj, x)
        candidate_add -= Dset
        candidate_add = sorted(candidate_add, key=lambda x: (-scores[x], x))

        for u in selected_vertices:
            T = set(Dset)
            T.remove(u)
            loss_T = current_loss(T)
            if loss_T < best_loss:
                best_loss = loss_T
                best_move = ("remove", u, None)

            for w in candidate_add:
                T = set(Dset)
                T.remove(u)
                T.add(w)
                loss_T = current_loss(T)
                if loss_T < best_loss:
                    best_loss = loss_T
                    best_move = ("swap", u, w)
                elif loss_T == best_loss and best_move is not None:
                    old_score = 0.0
                    if best_move[0] == "swap":
                        old_score = scores[best_move[2]] - scores[best_move[1]]
                    elif best_move[0] == "remove":
                        old_score = -scores[best_move[1]]
                    cand_score = scores[w] - scores[u]
                    if cand_score > old_score:
                        best_move = ("swap", u, w)

        if best_move is not None and best_loss < base_loss:
            kind, u, w = best_move
            if kind == "remove":
                Dset.remove(u)
                if verbose:
                    print(f"[LocalSwap][round {rd+1}] improve by remove {u}")
            else:
                Dset.remove(u)
                Dset.add(w)
                if verbose:
                    print(f"[LocalSwap][round {rd+1}] improve by swap {u} -> {w}")

            Dset = set(prune_redundant_vertices(adj, Dset, probs=scores, verbose=False))
            improved = True
            improved_any = True

        if verbose:
            print(f"[LocalSwap][round {rd+1}] size={len(Dset)}, improved={improved}")

        if not improved:
            break

    Dset = set(prune_redundant_vertices(adj, Dset, probs=scores, verbose=False))

    if verbose:
        print(f"[LocalSwap] final size={len(Dset)}, changed={improved_any}")

    return sorted(Dset)



def local_swap2_optimize_dominating_set(
    adj,
    D,
    probs=None,
    max_trials=200,
    verbose=False,
):
    """Try 2-for-1 local improvement while preserving domination."""
    import numpy as np

    scores = _probs_to_numpy_scores(probs) if probs is not None else np.zeros(len(adj), dtype=np.float32)
    Dset = set(D)

    if not check_domination(adj, Dset):
        if verbose:
            print("[LocalSwap2] input set is not a dominating set; skip swap-2.")
        return sorted(Dset)

    Dset = set(prune_redundant_vertices(adj, Dset, probs=scores, verbose=False))
    trials = 0

    while trials < max_trials:
        improved = False
        selected = sorted(Dset, key=lambda x: (scores[x], x))

        candidate_add = set()
        for u in Dset:
            candidate_add |= get_closed_neighborhood(adj, u)
            for x in adj[u]:
                candidate_add |= get_closed_neighborhood(adj, x)
        if not candidate_add:
            candidate_add = set(range(len(adj)))
        candidate_add -= Dset
        candidate_add = sorted(candidate_add, key=lambda x: (-scores[x], x))

        for i in range(len(selected)):
            if improved or trials >= max_trials:
                break
            for j in range(i + 1, len(selected)):
                if improved or trials >= max_trials:
                    break
                u1, u2 = selected[i], selected[j]
                base = set(Dset)
                base.remove(u1)
                base.remove(u2)
                for w in candidate_add:
                    trials += 1
                    cand = set(base)
                    cand.add(w)
                    if len(cand) >= len(Dset):
                        continue
                    if check_domination(adj, cand):
                        cand = set(prune_redundant_vertices(adj, cand, probs=scores, verbose=False))
                        if check_domination(adj, cand) and len(cand) < len(Dset):
                            if verbose:
                                print(f"[LocalSwap2] improve by removing {{{u1},{u2}}} and adding {w}: {len(Dset)} -> {len(cand)}")
                            Dset = cand
                            improved = True
                            break
        if not improved:
            break

    return sorted(Dset)

def ilp_polish_dominating_set(adj, D, time_limit=ILP_POLISH_TIME, verbose=ILP_POLISH_VERBOSE):
    """
    ILP-based polish / repair for dominating sets.

    Behavior:
    - If the input set D is already a dominating set, keep the ILP solution only
      when it is valid and no larger than D (polish mode).
    - If the input set D is not a dominating set, accept any valid ILP dominating
      set returned by the solver (repair mode).
    """
    try:
        import pulp
    except ImportError:
        if verbose:
            print("[ILP Polish] pulp not installed; skip polish/repair.")
        return sorted(D)

    N = len(adj)
    if N == 0:
        return []

    D0 = sorted(set(D))
    input_is_dom = check_domination(adj, D0)

    prob = pulp.LpProblem("DomSetPolish", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat=pulp.LpBinary) for i in range(N)]

    prob += pulp.lpSum(x[i] for i in range(N))

    for v in range(N):
        closed_nbhd = [v] + list(adj[v])
        prob += pulp.lpSum(x[u] for u in closed_nbhd) >= 1, f"dom_{v}"

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    if status not in ("Optimal", "Integer Feasible"):
        if verbose:
            print(f"[ILP Polish] status={status}; keep current solution.")
        return D0

    S = [i for i in range(N) if x[i].value() is not None and x[i].value() > 0.5]
    S = sorted(S)

    if not check_domination(adj, S):
        if verbose:
            print("[ILP Polish] solver returned invalid set; keep current solution.")
        return D0

    if input_is_dom:
        if len(S) <= len(D0):
            if verbose:
                print(f"[ILP Polish] improved {len(D0)} -> {len(S)} (status={status})")
            return S
        if verbose:
            print(f"[ILP Polish] no improvement ({len(D0)} -> {len(S)}); keep heuristic solution.")
        return D0

    if verbose:
        print(f"[ILP Repair] repaired invalid set {len(D0)} -> valid dominating set {len(S)} (status={status})")
    return S



def get_uncovered_hole_components(adj, D):
    """Return connected components of currently uncovered vertices."""
    undom = set(get_undominated_vertices(adj, D))
    comps = []
    while undom:
        s = undom.pop()
        comp = {s}
        stack = [s]
        while stack:
            v = stack.pop()
            for u in adj[v]:
                if u in undom:
                    undom.remove(u)
                    comp.add(u)
                    stack.append(u)
        comps.append(sorted(comp))
    comps.sort(key=lambda comp: (len(comp), comp[0] if comp else -1))
    return comps



def _expand_vertices_by_hops(adj, vertices, hops=1):
    frontier = set(vertices)
    seen = set(vertices)
    for _ in range(max(0, int(hops))):
        nxt = set()
        for v in frontier:
            nxt.update(adj[v])
        nxt -= seen
        if not nxt:
            break
        seen |= nxt
        frontier = nxt
    return seen


def build_conflict_region_for_hole(adj, current_set, hole_vertices, expand_hops=MICRO_ILP_EXPAND_HOPS, selected_hops=MICRO_ILP_CONFLICT_SELECTED_HOPS):
    """Build a local conflict region around one uncovered hole.

    The region contains the uncovered hole, its nearby vertices, and currently
    selected GNN vertices that may be replaced by a better local configuration.
    """
    hole = sorted(set(hole_vertices))
    if not hole:
        return [], []

    core = _expand_vertices_by_hops(adj, hole, hops=expand_hops)
    selected_nearby = {u for u in set(current_set) if u in _expand_vertices_by_hops(adj, hole, hops=selected_hops)}
    region = set(core) | selected_nearby
    for u in list(selected_nearby):
        region.update(adj[u])
    return sorted(region), hole


def micro_ilp_repair_conflict_region(adj, current_set, hole_vertices, probs=None, time_limit=MICRO_ILP_TIME_LIMIT, verbose=False):
    """Local adaptive ILP that may both add vertices and replace bad local GNN picks.

    Unlike the old micro-hole ILP, vertices inside the conflict region are not fixed.
    The solver can reselect the local configuration while keeping all outside-region
    selections fixed.
    """
    D0 = set(current_set)
    hole = sorted(set(hole_vertices))
    if not hole:
        return sorted(D0)

    still_uncovered = [v for v in hole if v in set(get_undominated_vertices(adj, D0))]
    if not still_uncovered:
        return sorted(D0)
    hole = still_uncovered

    region, _ = build_conflict_region_for_hole(adj, D0, hole)
    if not region:
        return sorted(D0)

    if len(hole) > MICRO_ILP_MAX_HOLE_SIZE or len(region) > MICRO_ILP_MAX_REGION_SIZE:
        if verbose:
            print(
                f"[Micro-ILP-Conflict] skip hole size={len(hole)}, region={len(region)} "
                f"(limits: hole<={MICRO_ILP_MAX_HOLE_SIZE}, region<={MICRO_ILP_MAX_REGION_SIZE})"
            )
        return sorted(D0)

    try:
        import pulp
    except ImportError:
        if verbose:
            print("[Micro-ILP-Conflict] pulp not installed; skip local repair.")
        return sorted(D0)

    scores = _probs_to_numpy_scores(probs) if probs is not None else None
    region_set = set(region)
    fixed_outside = D0 - region_set
    affected = _expand_vertices_by_hops(adj, region, hops=MICRO_ILP_AFFECTED_HOPS)

    prob = pulp.LpProblem("RepairConflictRegion", pulp.LpMinimize)
    x = {u: pulp.LpVariable(f"x_{u}", lowBound=0, upBound=1, cat=pulp.LpBinary) for u in region}

    eps = float(MICRO_ILP_OBJECTIVE_EPS)
    if scores is None:
        prob += pulp.lpSum(x[u] for u in region)
    else:
        prob += pulp.lpSum((1.0 + eps * (1.0 - float(scores[u]))) * x[u] for u in region)

    for v in affected:
        fixed_cover = 1 if (v in fixed_outside or any(u in fixed_outside for u in adj[v])) else 0
        local_vars = [x[u] for u in region if (u == v or u in adj[v])]
        if fixed_cover:
            continue
        if not local_vars:
            if verbose:
                print(f"[Micro-ILP-Conflict] affected vertex {v} has no local candidate; keep current solution.")
            return sorted(D0)
        prob += pulp.lpSum(local_vars) >= 1, f"dom_{v}"

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    if status not in ("Optimal", "Integer Feasible"):
        if verbose:
            print(f"[Micro-ILP-Conflict] status={status}; keep current solution.")
        return sorted(D0)

    chosen_region = {u for u in region if x[u].value() is not None and x[u].value() > 0.5}
    repaired = sorted(fixed_outside | chosen_region)
    if not check_domination(adj, repaired):
        if verbose:
            print("[Micro-ILP-Conflict] local solution invalid globally; keep current solution.")
        return sorted(D0)

    if verbose:
        removed_local = len(D0 & region_set) - len(chosen_region & D0)
        added_local = len(chosen_region - D0)
        print(
            f"[Micro-ILP-Conflict] hole={len(hole)} region={len(region)} removed={removed_local} "
            f"added={added_local} size={len(D0)} -> {len(repaired)}"
        )
    return repaired


def micro_ilp_repair_hole(adj, current_set, hole_vertices, time_limit=MICRO_ILP_TIME_LIMIT, verbose=False, probs=None):
    """Backward-compatible wrapper.

    If conflict-region mode is enabled, use adaptive local ILP that can replace
    weak local GNN vertices. Otherwise fall back to add-only local repair.
    """
    if MICRO_ILP_CONFLICT_ENABLE:
        return micro_ilp_repair_conflict_region(
            adj,
            current_set,
            hole_vertices,
            probs=probs,
            time_limit=time_limit,
            verbose=verbose,
        )

    D0 = set(current_set)
    hole = sorted(set(hole_vertices))
    if not hole:
        return sorted(D0)

    still_uncovered = [v for v in hole if v in set(get_undominated_vertices(adj, D0))]
    if not still_uncovered:
        return sorted(D0)
    hole = still_uncovered

    region = set(hole)
    for v in hole:
        region.update(adj[v])
    region = sorted(region)

    if len(hole) > MICRO_ILP_MAX_HOLE_SIZE or len(region) > MICRO_ILP_MAX_REGION_SIZE:
        if verbose:
            print(
                f"[Micro-ILP] skip hole size={len(hole)}, region={len(region)} "
                f"(limits: hole<={MICRO_ILP_MAX_HOLE_SIZE}, region<={MICRO_ILP_MAX_REGION_SIZE})"
            )
        return sorted(D0)

    try:
        import pulp
    except ImportError:
        if verbose:
            print("[Micro-ILP] pulp not installed; skip local hole repair.")
        return sorted(D0)

    candidates = [u for u in region if u not in D0]
    if not candidates:
        if verbose:
            print("[Micro-ILP] no local candidates available.")
        return sorted(D0)

    prob = pulp.LpProblem("RepairOneHole", pulp.LpMinimize)
    x = {u: pulp.LpVariable(f"x_{u}", lowBound=0, upBound=1, cat=pulp.LpBinary) for u in candidates}
    prob += pulp.lpSum(x[u] for u in candidates)

    for v in hole:
        repair_vars = [x[u] for u in candidates if (u == v or u in adj[v])]
        if not repair_vars:
            if verbose:
                print(f"[Micro-ILP] hole vertex {v} has no local repair candidate.")
            return sorted(D0)
        prob += pulp.lpSum(repair_vars) >= 1, f"cover_{v}"

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    if status not in ("Optimal", "Integer Feasible"):
        if verbose:
            print(f"[Micro-ILP] status={status}; keep current solution.")
        return sorted(D0)

    add_set = {u for u in candidates if x[u].value() is not None and x[u].value() > 0.5}
    repaired = sorted(D0 | add_set)
    if verbose:
        print(
            f"[Micro-ILP] hole={len(hole)} region={len(region)} add={len(add_set)} "
            f"size={len(D0)} -> {len(repaired)}"
        )
    return repaired



def ilp_repair_uncovered_vertices(adj, current_set, uncovered_vertices, time_limit=ILP_POLISH_TIME, verbose=False):
    """
    Add a minimum number of extra vertices so that all currently uncovered vertices
    become dominated. The current_set is kept fixed; this routine only adds vertices.
    It is an ILP over candidate vertices in the closed neighborhoods of uncovered vertices.
    """
    D0 = set(current_set)
    undom = sorted(set(uncovered_vertices))
    if not undom:
        return sorted(D0)

    try:
        import pulp
    except ImportError:
        if verbose:
            print("[ILP Repair-Uncovered] pulp not installed; skip.")
        return sorted(D0)

    candidate_set = set()
    for v in undom:
        candidate_set.add(v)
        candidate_set.update(adj[v])
    candidate_set -= D0
    candidates = sorted(candidate_set)

    if not candidates:
        if verbose:
            print("[ILP Repair-Uncovered] no candidate vertices available.")
        return sorted(D0)

    prob = pulp.LpProblem("RepairUncoveredVertices", pulp.LpMinimize)
    x = {u: pulp.LpVariable(f"x_{u}", lowBound=0, upBound=1, cat=pulp.LpBinary) for u in candidates}
    prob += pulp.lpSum(x[u] for u in candidates)

    for v in undom:
        repair_vars = [x[u] for u in candidates if (u == v or u in adj[v])]
        if not repair_vars:
            if verbose:
                print(f"[ILP Repair-Uncovered] vertex {v} has no repair candidate.")
            return sorted(D0)
        prob += pulp.lpSum(repair_vars) >= 1, f"cover_{v}"

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    if status not in ("Optimal", "Integer Feasible"):
        if verbose:
            print(f"[ILP Repair-Uncovered] status={status}; keep current solution.")
        return sorted(D0)

    add_set = {u for u in candidates if x[u].value() is not None and x[u].value() > 0.5}
    repaired = sorted(D0 | add_set)
    if verbose:
        print(f"[ILP Repair-Uncovered] added={len(add_set)}, size={len(D0)} -> {len(repaired)}")
    return repaired




def adaptive_subgraph_refine(adj_dict, selected_set, all_nodes=None, limit=50, probs=None, verbose=False):
    """Adaptive local ILP refinement over a conflict zone around all uncovered vertices.

    This pass does not freeze the locally selected GNN vertices. Instead, it
    builds an influence zone around the remaining uncovered vertices and lets
    ILP simultaneously add new vertices and replace weak local picks, while
    keeping selections outside the zone fixed.
    """
    D0 = set(selected_set)
    N = len(adj_dict) if all_nodes is None else len(list(all_nodes))
    if N == 0:
        return sorted(D0)

    uncovered = set(get_undominated_vertices(adj_dict, D0))
    if not uncovered:
        return sorted(D0)

    influence_zone = set(uncovered)
    core = _expand_vertices_by_hops(adj_dict, uncovered, hops=ADAPTIVE_SUBGRAPH_REFINE_EXPAND_HOPS)
    influence_zone |= core
    for u in list(core):
        influence_zone.update(adj_dict[u])

    selected_near = {u for u in D0 if u in influence_zone}
    for u in list(selected_near):
        influence_zone.update(adj_dict[u])

    if len(influence_zone) > max(1, int(limit)):
        hot = set()
        scores = _probs_to_numpy_scores(probs) if probs is not None else None
        cand = list(influence_zone)
        if scores is not None:
            cand.sort(key=lambda u: (u not in D0, -float(scores[u]), u))
        else:
            cand.sort(key=lambda u: (u not in D0, -len(adj_dict[u]), u))
        hot.update(sorted(uncovered))
        hot.update(selected_near)
        for u in cand:
            hot.add(u)
            if len(hot) >= int(limit):
                break
        influence_zone = hot

    region = sorted(influence_zone)
    region_set = set(region)
    if len(region) > ADAPTIVE_SUBGRAPH_REFINE_MAX_REGION_SIZE:
        if verbose:
            print(f"[Adaptive-Refine] skip region={len(region)} > {ADAPTIVE_SUBGRAPH_REFINE_MAX_REGION_SIZE}")
        return sorted(D0)

    fixed_outside = D0 - region_set
    affected = _expand_vertices_by_hops(adj_dict, region, hops=ADAPTIVE_SUBGRAPH_REFINE_AFFECTED_HOPS)

    try:
        import pulp
    except ImportError:
        if verbose:
            print("[Adaptive-Refine] pulp not installed; skip adaptive refinement.")
        return sorted(D0)

    scores = _probs_to_numpy_scores(probs) if probs is not None else None
    prob = pulp.LpProblem("AdaptiveSubgraphRefine", pulp.LpMinimize)
    x = {u: pulp.LpVariable(f"x_{u}", lowBound=0, upBound=1, cat=pulp.LpBinary) for u in region}
    eps = float(MICRO_ILP_OBJECTIVE_EPS)
    if scores is None:
        prob += pulp.lpSum(x[u] for u in region)
    else:
        prob += pulp.lpSum((1.0 + eps * (1.0 - float(scores[u]))) * x[u] for u in region)

    for v in affected:
        fixed_cover = 1 if (v in fixed_outside or any(u in fixed_outside for u in adj_dict[v])) else 0
        if fixed_cover:
            continue
        local_vars = [x[u] for u in region if (u == v or u in adj_dict[v])]
        if not local_vars:
            if verbose:
                print(f"[Adaptive-Refine] affected vertex {v} has no local candidate; keep current solution.")
            return sorted(D0)
        prob += pulp.lpSum(local_vars) >= 1, f"dom_{v}"

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=ADAPTIVE_SUBGRAPH_REFINE_TIME_LIMIT)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    if status not in ("Optimal", "Integer Feasible"):
        if verbose:
            print(f"[Adaptive-Refine] status={status}; keep current solution.")
        return sorted(D0)

    chosen_region = {u for u in region if x[u].value() is not None and x[u].value() > 0.5}
    repaired = sorted(fixed_outside | chosen_region)
    if not check_domination(adj_dict, repaired):
        if verbose:
            print("[Adaptive-Refine] local solution invalid globally; keep current solution.")
        return sorted(D0)

    if len(repaired) > len(D0):
        # Only accept a larger solution when it strictly reduces uncovered vertices.
        new_undom = len(get_undominated_vertices(adj_dict, repaired))
        old_undom = len(uncovered)
        if new_undom >= old_undom:
            if verbose:
                print(f"[Adaptive-Refine] reject non-improving larger solution {len(D0)} -> {len(repaired)}")
            return sorted(D0)

    if verbose:
        removed_local = len((D0 & region_set) - chosen_region)
        added_local = len(chosen_region - D0)
        print(
            f"[Adaptive-Refine] region={len(region)} affected={len(affected)} "
            f"removed={removed_local} added={added_local} size={len(D0)} -> {len(repaired)}"
        )
    return repaired


def gnn_raw_then_complete(
    adj,
    probs,
    threshold=PROB_SOFTREPAIR_THRESHOLD,
    ilp_cutoff=ILP_CompleteTime,
    beta=PROB_SOFTREPAIR_BETA,
    verbose=PROB_SOFTREPAIR_VERBOSE,
    completion_mode=None,
):
    """
    Coverage-aware raw -> prune -> repair pipeline.

    v24.2 patch:
    1. stage-by-stage stats are recorded in gnn_raw_then_complete.last_stats
    2. high-coverage seeds prefer localized repair only
    3. repair acceptance is size-sensitive, especially near domination
    4. heavy beam/greedy repair is skipped when the remaining uncovered part is small
    """
    N = len(adj)
    if N == 0:
        gnn_raw_then_complete.last_stats = {"stages": []}
        return []

    scores = _probs_to_numpy_scores(probs)
    if len(scores) != N:
        raise ValueError(f"[gnn_raw_then_complete] len(scores)={len(scores)} != N={N}")

    if verbose:
        print("[Pipeline] strategy = adaptive-raw -> prune -> micro-ILP -> adaptive-subgraph-ILP -> selective beam/greedy -> prune/local")

    stage_stats = []

    def _record_stage(name, S):
        S = set(S)
        stat = {
            "stage": name,
            "size": len(S),
            "cov": float(domination_coverage(adj, S)),
            "undom": int(len(get_undominated_vertices(adj, S))),
        }
        stage_stats.append(stat)
        if verbose:
            print(f"[Stage] {name}: size={stat['size']} cov={stat['cov']:.4f} undom={stat['undom']}")
        return stat

    S_raw = set(gnn_raw_only(adj, scores, threshold=threshold))
    _record_stage("raw", S_raw)

    S_prune = set(prune_redundant_vertices(adj, S_raw, probs=scores, verbose=verbose))
    if not S_prune:
        best = int(scores.argmax())
        S_prune = {best}
        if verbose:
            print(f"[Pipeline] pruned to empty set, re-seed with argmax vertex {best}.")

    best_set = set(S_prune)
    best_stat = _record_stage("prune", best_set)
    best_undom = best_stat["undom"]

    def maybe_accept(candidate, stage_name):
        nonlocal best_set, best_undom, best_stat
        cand = set(prune_redundant_vertices(adj, candidate, probs=scores, verbose=False))
        cand_undom = len(get_undominated_vertices(adj, cand))
        cand_size = len(cand)
        cand_cov = domination_coverage(adj, cand)
        cur_size = len(best_set)
        cur_undom = best_undom
        cur_cov = best_stat["cov"] if best_stat else domination_coverage(adj, best_set)

        near_dom = (
            cur_cov >= max(PROB_SOFTREPAIR_TARGET_RAW_COVERAGE, 0.97)
            and cur_undom <= max(PROB_LOCAL_REPAIR_MAX_UNDOM, int(0.03 * N))
        )
        strict_add_k = PROB_SOFTREPAIR_STRICT_ACCEPT_ADD_K
        strict_ratio_cap = int(math.ceil(PROB_SOFTREPAIR_ACCEPT_WORSE_RATIO * max(cur_size, 1)))
        strict_size_cap = min(cur_size + strict_add_k, strict_ratio_cap)
        soft_size_cap = min(cur_size + max(strict_add_k, int(math.ceil(0.02 * N))), int(math.ceil(PROB_SOFTREPAIR_ACCEPT_WORSE_RATIO * max(cur_size, 1))))

        accept = False
        reason = ""
        if cand_undom == 0 and cur_undom > 0:
            accept = True
            reason = "full-dom always accept"
        elif cand_undom < cur_undom:
            cap = strict_size_cap if near_dom else soft_size_cap
            accept = cand_size <= cap
            reason = f"better-undom cap={cap}"
        elif cand_undom == cur_undom and cand_size < cur_size:
            accept = True
            reason = "same-undom smaller-size"
        elif cand_undom == cur_undom and cand_size == cur_size and cand_cov >= cur_cov:
            accept = True
            reason = "same-size no-worse-cov"

        if accept:
            if verbose:
                print(
                    f"[Pipeline] accept {stage_name}: size {cur_size}->{cand_size}, "
                    f"undom {cur_undom}->{cand_undom}, cov={cand_cov:.4f}, reason={reason}"
                )
            best_set = cand
            best_undom = cand_undom
            best_stat = _record_stage(stage_name, best_set)
        else:
            if verbose:
                print(
                    f"[Pipeline] reject {stage_name}: size {cur_size}->{cand_size}, "
                    f"undom {cur_undom}->{cand_undom}, cov={cand_cov:.4f}, reason={reason or 'size/cov gate'}"
                )
        return accept

    if best_undom > 0 and MICRO_ILP_ENABLE:
        try:
            hole_components = get_uncovered_hole_components(adj, best_set)
            if verbose:
                hole_sizes = [len(comp) for comp in hole_components]
                print(f"[Pipeline] uncovered holes={len(hole_components)}, sizes={hole_sizes[:12]}")
            for hole_idx, hole in enumerate(hole_components, 1):
                cur_undom_set = set(get_undominated_vertices(adj, best_set))
                active_hole = [v for v in hole if v in cur_undom_set]
                if not active_hole:
                    continue
                repaired = micro_ilp_repair_hole(
                    adj,
                    best_set,
                    active_hole,
                    time_limit=min(ilp_cutoff, MICRO_ILP_TIME_LIMIT),
                    verbose=verbose or ILP_POLISH_VERBOSE,
                    probs=scores,
                )
                maybe_accept(repaired, f"micro-ILP#{hole_idx}")
                if best_undom == 0:
                    break
        except Exception as e:
            if verbose:
                print(f"[Pipeline] micro-ILP hole repair failed: {e}")

    if best_undom > 0 and ADAPTIVE_SUBGRAPH_REFINE_ENABLE:
        try:
            refined = adaptive_subgraph_refine(
                adj,
                best_set,
                all_nodes=range(N),
                limit=ADAPTIVE_SUBGRAPH_REFINE_MAX_REGION_SIZE,
                probs=scores,
                verbose=verbose or ILP_POLISH_VERBOSE,
            )
            maybe_accept(refined, "adaptive-subgraph-ILP")
        except Exception as e:
            if verbose:
                print(f"[Pipeline] adaptive-subgraph refine failed: {e}")

    current_cov = domination_coverage(adj, best_set)
    local_only_mode = (best_undom > 0) and (
        (current_cov >= max(PROB_SOFTREPAIR_TARGET_RAW_COVERAGE, 0.97))
        and (best_undom <= max(PROB_LOCAL_REPAIR_MAX_UNDOM, int(0.03 * N)))
    )
    need_heavy_repair = (best_undom > 0) and (not local_only_mode) and (current_cov < PROB_SOFTREPAIR_SKIP_HEAVY_REPAIR_IF_NEAR_DOM)
    if best_undom > 0 and local_only_mode and verbose:
        print(
            f"[Pipeline] local-only repair: cov={current_cov:.4f}, undom={best_undom}; skip beam/global greedy."
        )
    elif best_undom > 0 and not need_heavy_repair and verbose:
        print(
            f"[Pipeline] skip heavy repair: cov={current_cov:.4f} >= "
            f"{PROB_SOFTREPAIR_SKIP_HEAVY_REPAIR_IF_NEAR_DOM:.4f}; keep selective local polish only."
        )

    mode = _resolve_completion_mode(completion_mode)
    if best_undom > 0 and need_heavy_repair and mode == 'beam':
        beam_set = beam_search_completion_with_initial(adj, best_set, probs=scores, rl_probs=None, verbose=verbose)
        maybe_accept(beam_set, "beam")

    if best_undom > 0 and need_heavy_repair:
        greedy_set = guided_greedy_completion_with_initial(adj, best_set, probs=scores, verbose=verbose)
        maybe_accept(greedy_set, "guided-greedy")

    S_final = set(prune_redundant_vertices(adj, best_set, probs=scores, verbose=False))
    _record_stage("pre-final-prune", S_final)
    S_final = set(local_swap_optimize_dominating_set(adj, S_final, probs=scores, verbose=verbose))
    _record_stage("local-swap", S_final)
    if LOCAL_SWAP2_ENABLE:
        S_final = set(local_swap2_optimize_dominating_set(
            adj, S_final, probs=scores, max_trials=LOCAL_SWAP2_MAX_TRIALS, verbose=verbose
        ))
        _record_stage("local-swap2", S_final)
    S_final = set(prune_redundant_vertices(adj, S_final, probs=scores, verbose=False))

    if not check_domination(adj, S_final):
        fallback = guided_greedy_completion_with_initial(adj, S_final, probs=scores, verbose=verbose)
        fallback = set(prune_redundant_vertices(adj, fallback, probs=scores, verbose=False))
        fallback_undom = len(get_undominated_vertices(adj, fallback))
        fallback_size = len(fallback)
        final_size = len(S_final)
        if fallback_undom == 0 and fallback_size <= final_size + PROB_SOFTREPAIR_STRICT_ACCEPT_ADD_K:
            S_final = fallback
            if verbose:
                print(f"[Pipeline] accept fallback domination repair: size {final_size}->{fallback_size}")
        elif verbose:
            print(f"[Pipeline] reject fallback domination repair: size {final_size}->{fallback_size}, undom={fallback_undom}")

    final_stat = _record_stage("final", S_final)
    gnn_raw_then_complete.last_stats = {
        "stages": stage_stats,
        "raw_size": len(S_raw),
        "raw_cov": float(stage_stats[0]['cov']) if stage_stats else 0.0,
        "final_size": len(S_final),
        "final_cov": float(final_stat['cov']),
        "final_undom": int(final_stat['undom']),
        "heavy_repair_used": bool(need_heavy_repair),
        "local_only_mode": bool(local_only_mode),
    }
    return sorted(S_final)
def save_training_graphs_safe(folder, graphs):
    """
    儲存訓練資料（模型無關，共用 trainset）。
    固定檔名：train_safe_graphs.pt
    """
    from pathlib import Path
    import torch

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    fname = folder / "train_safe_graphs.pt"

    payload = {
        "graphs": graphs,
        "num_graphs": len(graphs)
    }

    try:
        torch.save(payload, str(fname))
        print(f"[TrainSafe] Saved: {fname} (graphs={len(graphs)})")
    except Exception as e:
        print(f"[TrainSafe] Save failed: {e}")

def load_training_graphs_safe(folder):
    from pathlib import Path
    import torch

    folder = Path(folder)
    fname = folder / "train_safe_graphs.pt"
    if not fname.is_file():
        print(f"[TrainSafe] File not found: {fname}")
        return []

    try:
        # ✅ PyTorch 新版：請明確指定 weights_only=False
        payload = torch.load(str(fname), map_location="cpu", weights_only=False)
    except TypeError:
        # 舊版 torch 沒有 weights_only 參數
        payload = torch.load(str(fname), map_location="cpu")
    except Exception as e:
        print(f"[TrainSafe] Failed to load: {fname} ({e})")
        return []

    # 兼容 dict / list
    if isinstance(payload, dict):
        graphs = payload.get("graphs", payload)
    else:
        graphs = payload

    if not graphs:
        print("[TrainSafe] Empty graphs in file.")
        return []
    
    # ✅ 加這行
    print(f"[TrainSafe] Loaded {len(graphs)} graphs from {fname}")
    return graphs

def analyze_train_safe_Connected_folder(folder):
    graphs = load_training_graphs_safe(folder)
    if graphs is None:
        return {"error": "no file"}
    dims = set()
    for g in graphs:
        data = g["data"]
        dims.add(data.x.shape[1])
    edge_index_shapes = set()
    edge_directions = set()
    
    edge_ok = True
    for g in graphs:
        data = g["data"]
        if (not hasattr(data, "edge_index")) or data.edge_index is None:
            edge_ok = False
            break

    rep = {
        "unique_dims": sorted(list(dims)),
        "health": (len(dims) == 1 and bad == 0),
        "bad_graphs": bad,
        "edge_health": edge_ok,
    }
    return rep

# =========================================================
#  訓練資料產生建議預設值 & 輔助輸入函式（exp 版）
# =========================================================

def input_int_with_default(prompt: str, default: int, min_value=None, max_value=None) -> int:
    while True:
        s = input(f"{prompt} [預設 {default}]：").strip()
        if s == "":
            value = default
        else:
            try:
                value = int(s)
            except ValueError:
                print("❌ 輸入錯誤，請輸入整數。")
                continue
        if min_value is not None and value < min_value:
            print(f"❌ 輸入值不可小於 {min_value}。")
            continue
        if max_value is not None and value > max_value:
            print(f"❌ 輸入值不可大於 {max_value}。")
            continue
        return value


def input_float_with_default(prompt: str, default: float, min_value=None, max_value=None) -> float:
    while True:
        s = input(f"{prompt} [預設 {default}]：").strip()
        if s == "":
            value = default
        else:
            try:
                value = float(s)
            except ValueError:
                print("❌ 輸入錯誤，請輸入數字，例如 0.2。")
                continue
        if min_value is not None and value < min_value:
            print(f"❌ 輸入值不可小於 {min_value}。")
            continue
        if max_value is not None and value > max_value:
            print(f"❌ 輸入值不可大於 {max_value}。")
            continue
        return value

# =========================================================
#  功能 1：產生訓練資料（Irregular Grid / Supergrid）
# =========================================================

def generate_training_data():
    print("=== 產生『非矩形』訓練資料 (ILP + Greedy + Boundary-Aware Features + PE + RWE) ===")
    print("規則：輸入 m, n, N，並依 hole rate p ∈ {0,0.2,0.4,0.5,0.6,0.8} 平均產生 N/6 個圖（若 N 不能被 6 整除，餘數會依序分配到較小的 p）。")

    m = input_int_with_default("m (列數)", TRAIN_DEFAULT_MIN_M)
    n = input_int_with_default("n (行數)", TRAIN_DEFAULT_MIN_N)
    num = input_int_with_default("總訓練圖數量 N", TRAIN_DEFAULT_NUM)

    hole_rates = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8]
    base = num // len(hole_rates)
    rem = num % len(hole_rates)
    per_counts = [base + (1 if i < rem else 0) for i in range(len(hole_rates))]

    print(f"[TrainSet] m={m}, n={n}, N={num}")
    print("[TrainSet] hole-rate 分配：", {hole_rates[i]: per_counts[i] for i in range(len(hole_rates))})

    graphs = []
    reset_ilp_profile_stats()
    print("[建立『非矩形』訓練資料 ...]")
    print(f"[TrainSet] ILP adaptive teacher: enabled={ILP_ADAPTIVE_TEACHER}, hard_cap={ILP_TRAIN_MAX_NODES}, min_success_rate={ILP_MIN_SUCCESS_RATE:.0%}, max_avg_time={ILP_MAX_AVG_TIME:.1f}s, warmup={ILP_MIN_PROFILE_ATTEMPTS}")

    for p, cnt in zip(hole_rates, per_counts):
        for _ in range(cnt):
            adj, coords, actual_hole = build_irregular_grid_adj(
                m, n, hole_ratio=float(p), ensure_connected=True
            )
            data = build_full_features(m, n, adj, coords=coords, pe_dim=8, rwe_dim=16)

            N_nodes = len(adj)
            sol_ilp, sol_gr, teacher_info = choose_teacher_solutions_for_training(adj, N_nodes)

            y_ilp = torch.zeros(N_nodes)
            y_ilp[sol_ilp] = 1
            y_gr = torch.zeros(N_nodes)
            y_gr[sol_gr] = 1

            if not hasattr(data, "x") or data.x is None:
                raise RuntimeError("build_full_features produced data without x")

            graphs.append({
                "m": int(m), "n": int(n),
                "adj": adj,
                "coords": coords,
                "data": data,
                "labels_ilp": y_ilp,
                "labels_greedy": y_gr,
                "teacher_mode": teacher_info.get("teacher_mode", "Unknown"),
                "teacher_reason": teacher_info.get("reason", ""),
                "ilp_attempted": teacher_info.get("ilp_attempted", False),
                "ilp_success": teacher_info.get("ilp_success", False),
                "ilp_time_sec": teacher_info.get("ilp_time_sec", 0.0),
                "ilp_status": teacher_info.get("ilp_status", "UNKNOWN"),
                "ilp_status_class": teacher_info.get("ilp_status_class", _normalize_ilp_status(teacher_info.get("ilp_status", "UNKNOWN"))),
                "ilp_optimal": teacher_info.get("ilp_optimal", False),
                "hole_ratio": float(p),
                "actual_hole": float(actual_hole),
                "dynamic_ilp_weight": float(get_dynamic_teacher_weights(N_nodes)[0]),
                "dynamic_greedy_weight": float(get_dynamic_teacher_weights(N_nodes)[1]),
            })

    # 資料夾命名：固定 m,n 並標示多 hole rates
    hr_tag = "-".join([str(p).rstrip("0").rstrip(".") if p != 0 else "0" for p in hole_rates])
    folder = TRAINPATH / f"Train_{GRAPH_TOPOLOGY}_m{m}-n{n}-N{num}_HoleRate[{hr_tag}]"
    save_training_graphs_safe(folder, graphs)
    print_ilp_profile_summary()

    rep = analyze_train_safe_Connected_folder(folder)
    if "error" in rep:
        print(f"[TrainSafe] 新資料夾分析錯誤: {rep['error']}")
    else:
        print("[TrainSafe] 新資料夾特徵維度：", rep["unique_dims"])
        print("[TrainSafe] feature_dim 健康：", rep["health"])
        print("[TrainSafe] edge_index 健康：", rep["edge_health"])

# =========================================================
#  功能 A：產生測試資料（Irregular Grid / Supergrid）
#  - 只存「圖本身」(adj/coords/data)，不存 labels
# =========================================================


def save_test_graphs_safe(graphs, folder: Path):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fname = folder / "test_graphs.pt"
    torch.save(graphs, str(fname))
    print(f"[TestSet] Saved {len(graphs)} graphs to {fname}")

def load_test_graphs_safe(folder: str):
    fname = os.path.join(str(folder), "test_graphs.pt")
    if not os.path.exists(fname):
        print(f"[TestSet] File not found: {fname}")
        return None
    try:
        graphs = torch.load(fname, map_location="cpu", weights_only=False)
    except TypeError:
        graphs = torch.load(fname, map_location="cpu")
    print(f"[TestSet] Loaded {len(graphs)} graphs from {fname}")
    return graphs

def select_test_folder_gui():
    try:
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(initialdir=str(TESTPATH), title="Select TestSet Folder")
        root.destroy()
        if folder:
            return folder
        return None
    except Exception:
        # fallback: CLI
        s = input("請輸入 TestSet 資料夾完整路徑：").strip()
        return s or None

def generate_test_data():
    print("=== 產生『測試』資料集 (Irregular Grid/Supergrid) ===")
    print("規則：輸入 m, n, N，並依 hole rate p ∈ {0,0.2,0.4,0.5,0.6,0.8} 平均產生 N/6 個圖（若 N 不能被 6 整除，餘數會依序分配到較小的 p）。")

    m = input_int_with_default("m (列數)", TEST_DEFAULT_MIN_M)
    n = input_int_with_default("n (行數)", TEST_DEFAULT_MIN_N)
    num = input_int_with_default("總測試圖數量 N", TEST_DEFAULT_NUM)

    hole_rates = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8]
    base = num // len(hole_rates)
    rem = num % len(hole_rates)
    per_counts = [base + (1 if i < rem else 0) for i in range(len(hole_rates))]

    print(f"[TestSet] m={m}, n={n}, N={num}")
    print("[TestSet] hole-rate 分配：", {hole_rates[i]: per_counts[i] for i in range(len(hole_rates))})

    graphs = []
    print("[建立『非矩形』測試資料 ...]")

    for p, cnt in zip(hole_rates, per_counts):
        for _ in range(cnt):
            adj, coords, actual_hole = build_irregular_grid_adj(
                m, n, hole_ratio=float(p), ensure_connected=True
            )

            n_nodes = len(adj)
            if n_nodes <= 0:
                print(f"[TestSet] skip empty graph: m={m}, n={n}, hole={float(p):.2f}")
                continue

            data = build_full_features(m, n, adj, coords=coords, pe_dim=8, rwe_dim=16)
            ilp_w, greedy_w = get_dynamic_teacher_weights(n_nodes)

            graphs.append({
                "m": int(m),
                "n": int(n),
                "N_nodes": int(n_nodes),
                "adj": adj,
                "coords": coords,
                "data": data,
                "hole_ratio": float(p),
                "actual_hole": float(actual_hole),
                "dynamic_ilp_weight": float(ilp_w),
                "dynamic_greedy_weight": float(greedy_w),
            })

    hr_tag = "-".join([str(p).rstrip("0").rstrip(".") if p != 0 else "0" for p in hole_rates])
    folder = TESTPATH / f"Test_{GRAPH_TOPOLOGY}_m{m}-n{n}-N{num}_HoleRate[{hr_tag}]"
    save_test_graphs_safe(graphs, folder)

# =========================================================
#  功能 B：在測試資料集上一次跑 GNN + Greedy + ILP 並輸出 CSV
# =========================================================

def _infer_model_in_dim(model: nn.Module) -> int:
    """Infer the node feature input dimension from common GNN module layouts."""
    # 1) Explicit input encoder used by Transformer / GPS-style models
    for name in ["node_encoder", "input_proj", "in_proj", "encoder"]:
        if hasattr(model, name):
            mod = getattr(model, name)
            if hasattr(mod, "in_features"):
                return int(mod.in_features)
            if hasattr(mod, "weight"):
                return int(mod.weight.shape[1])

    # 2) PyG conv stacks such as GCNConv/SAGEConv/GATv2Conv/TransformerConv
    if hasattr(model, "convs") and len(getattr(model, "convs", [])) > 0:
        conv0 = model.convs[0]
        for attr in ["lin", "lin_l", "lin_src"]:
            if hasattr(conv0, attr):
                sub = getattr(conv0, attr)
                if hasattr(sub, "weight"):
                    return int(sub.weight.shape[1])
        if hasattr(conv0, "in_channels") and isinstance(conv0.in_channels, int):
            return int(conv0.in_channels)

    # 3) Other common first-layer names
    for name in ["conv1", "lin1", "lin_in", "input_layer"]:
        if hasattr(model, name):
            mod = getattr(model, name)
            if hasattr(mod, "in_features"):
                return int(mod.in_features)
            if hasattr(mod, "weight"):
                return int(mod.weight.shape[1])
            if hasattr(mod, "in_channels") and isinstance(mod.in_channels, int):
                return int(mod.in_channels)

    raise RuntimeError(f"Cannot infer model input dimension for model type {type(model).__name__}.")

"""
def _pad_or_trunc_x(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    cur = x.size(1)
    if cur == target_dim:
        return x
    if cur < target_dim:
        pad = torch.zeros((x.size(0), target_dim - cur), dtype=x.dtype)
        return torch.cat([x, pad], dim=1)
    # cur > target_dim: truncate
    return x[:, :target_dim]
"""
def _pad_or_trunc_x(x, target_dim):
    """
    將節點特徵 x 補到 target_dim 或截斷到 target_dim。
    保持與原 tensor 相同的 device / dtype。
    """
    if x is None:
        raise ValueError("x is None in _pad_or_trunc_x")

    cur_dim = x.size(1)

    if cur_dim == target_dim:
        return x

    if cur_dim > target_dim:
        return x[:, :target_dim]

    # cur_dim < target_dim → padding
    pad_dim = target_dim - cur_dim
    pad = torch.zeros(
        (x.size(0), pad_dim),
        dtype=x.dtype,
        device=x.device
    )
    return torch.cat([x, pad], dim=1)
    
def _append_zero_state_features(x: torch.Tensor, extra_dim: int = RL_STATE_EXTRA_DIM) -> torch.Tensor:
    if extra_dim <= 0:
        return x
    pad = torch.zeros((x.size(0), extra_dim), dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=1)

def _prepare_static_input_with_state_slots(x: torch.Tensor, static_dim: int, extra_dim: int = RL_STATE_EXTRA_DIM) -> torch.Tensor:
    x = _pad_or_trunc_x(x, static_dim)
    return _append_zero_state_features(x, extra_dim=extra_dim)

def _precompute_closed_neighborhoods(adj):
    closed_nbs = []
    for v, nbrs in enumerate(adj):
        cur = {int(v)}
        for u in nbrs:
            cur.add(int(u))
        closed_nbs.append(sorted(cur))
    return closed_nbs

def _build_rl_state_features(adj, selected, dominated, step_idx, closed_nbs=None):
    N = len(adj)
    device = selected.device
    dtype = torch.float32

    if closed_nbs is None:
        closed_nbs = _precompute_closed_neighborhoods(adj)

    undominated = ~dominated
    residual = torch.zeros(N, dtype=dtype, device=device)
    marginal = torch.zeros(N, dtype=dtype, device=device)

    for v in range(N):
        nb = closed_nbs[v]
        nb_idx = torch.tensor(nb, dtype=torch.long, device=device)
        undom_count = undominated[nb_idx].sum().float()
        denom = float(max(len(nb), 1))
        residual[v] = undom_count / denom
        marginal[v] = undom_count / float(max(N, 1))

    step_value = float(step_idx) / float(max(N, 1))
    step_feat = torch.full((N,), step_value, dtype=dtype, device=device)

    return torch.stack([
        selected.float(),
        dominated.float(),
        residual,
        marginal,
        step_feat,
    ], dim=1)

def _compose_rl_model_input(x_with_slots: torch.Tensor, adj, selected, dominated, step_idx, closed_nbs=None, extra_dim: int = RL_STATE_EXTRA_DIM) -> torch.Tensor:
    if extra_dim <= 0:
        return x_with_slots
    if x_with_slots.size(1) < extra_dim:
        raise RuntimeError(f"[RL] input feature dim {x_with_slots.size(1)} < extra_dim {extra_dim}")
    static_x = x_with_slots[:, :-extra_dim]
    state_x = _build_rl_state_features(adj, selected, dominated, step_idx, closed_nbs=closed_nbs).to(dtype=x_with_slots.dtype)
    return torch.cat([static_x, state_x], dim=1)

def run_testset_gnn_greedy_ilp():
    global GLOBAL_MODELS, GLOBAL_DEVICE
    if not GLOBAL_MODELS or GLOBAL_DEVICE is None:
        print("❌ 尚未訓練/載入模型。請先執行選單 5 或 6。")
        return

    model_name = list(GLOBAL_MODELS.keys())[0]
    model = GLOBAL_MODELS[model_name]
    model.eval()

    folder = select_test_folder_gui()
    if not folder:
        print("❌ 未選擇 TestSet 資料夾。")
        return

    graphs = load_test_graphs_safe(folder)
    if not graphs:
        print("❌ TestSet 空或讀取失敗。")
        return

    # 由載入模型時記錄的 in_dim（最穩）；若沒有就從 model.state_dict() 推回
    in_dim = GLOBAL_MODEL_IN_DIM or infer_in_dim_from_state_dict(SELECTED_GNN, model.state_dict())
    print(f"[TestSet] Using model={model_name}, model_in_dim={in_dim}")

    # 對齊 TestSet 的 feature_dim（pad / trunc）以符合模型輸入
    for g in graphs:
        x = g["data"].x
        if x.size(1) < in_dim:
            g["data"].x = pad_features_to_dim(x, in_dim)
        elif x.size(1) > in_dim:
            g["data"].x = x[:, :in_dim]

    os.makedirs(EXPERIMENT_RESULTS_DIR, exist_ok=True)
    import datetime, csv
    folder_name = Path(folder).name   # 只取最後一層名稱
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(EXPERIMENT_RESULTS_DIR, f"testset_eval_4methods_v38_{model_name}-{folder_name}-{timestamp}.csv")
    summary_csv_path = os.path.join(EXPERIMENT_RESULTS_DIR, f"testset_summary_v38_{model_name}-{folder_name}-{timestamp}.csv")

    header = [
        "idx", "m", "n", "N_nodes", "hole_ratio", "topology", "method", "set_size", "coverage", "time_sec",
        "raw_size", "added", "removed", "completion_rate", "gnn_contribution", "raw_coverage",
        "ilp_status", "ilp_status_class", "ilp_optimal", "ilp_success", "ilp_fallback",
        "ilp_attempted", "ilp_time_sec", "ilp_gap_ref", "is_valid_dom"
    ]

    def _coverage_local(adj_local, S):
        return compute_coverage(adj_local, S)

    def _is_valid_dom_local(adj_local, S):
        return int(len(get_undominated_vertices(adj_local, S)) == 0)

    def _write_row(
        wr, *, idx, m, n, n_nodes, hole, adj_local, method, S, time_sec,
        raw_size="", added="", removed="", completion_rate="", gnn_contribution="", raw_coverage="",
        ilp_status="", ilp_status_class="", ilp_optimal="", ilp_success="", ilp_fallback="",
        ilp_attempted="", ilp_time_sec="", ilp_gap_ref=""
    ):
        wr.writerow([
            idx, m, n, n_nodes, hole, GRAPH_TOPOLOGY, method,
            len(S), _coverage_local(adj_local, S), time_sec,
            raw_size, added, removed, completion_rate, gnn_contribution, raw_coverage,
            ilp_status, ilp_status_class, ilp_optimal, ilp_success, ilp_fallback,
            ilp_attempted, ilp_time_sec, ilp_gap_ref, _is_valid_dom_local(adj_local, S)
        ])

    method_sizes = defaultdict(list)
    method_covs = defaultdict(list)
    ilp_opt_sizes = []
    ilp_feas_sizes = []
    ilp_all_sizes = []
    ilp_counts = {"optimal": 0, "feasible_nonoptimal": 0, "failed_or_missing": 0}

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(header)

        for i, g in enumerate(graphs):
            m = g["m"]; n = g["n"]
            adj = g["adj"]
            hole = float(g.get("actual_hole", g.get("hole_ratio", 0.0)))
            data = g["data"].to(GLOBAL_DEVICE)

            # make feature dim match model
            data.x = _pad_or_trunc_x(data.x, in_dim).to(GLOBAL_DEVICE)

            # ---- ILP baseline with info ----
            t0 = time.perf_counter()
            S_ilp, ilp_info = ilp_minimum_dominating_set_with_info(adj, time_limit=ILP_MaxTime)
            t_ilp_outer = time.perf_counter() - t0
            ilp_status = str(ilp_info.get("status", "UNKNOWN"))
            ilp_status_class = str(ilp_info.get("status_class", _normalize_ilp_status(ilp_status)))
            ilp_optimal = bool(ilp_info.get("optimal", False))
            ilp_success = bool(ilp_info.get("success", False))
            ilp_fallback = bool(ilp_info.get("fallback", False))
            ilp_attempted = bool(ilp_info.get("attempted", False))
            ilp_time_sec = float(ilp_info.get("time_sec", t_ilp_outer))

            if ilp_optimal:
                ilp_gap_ref = "OPTIMAL"
                ilp_counts["optimal"] += 1
                ilp_opt_sizes.append(len(S_ilp))
            elif ilp_status_class == "TIME_LIMIT_FEASIBLE":
                ilp_gap_ref = "LOWER_BOUND"
                ilp_counts["feasible_nonoptimal"] += 1
                ilp_feas_sizes.append(len(S_ilp))
            else:
                ilp_gap_ref = "FALLBACK_OR_FAILED"
                ilp_counts["failed_or_missing"] += 1
            ilp_all_sizes.append(len(S_ilp))

            _write_row(
                wr,
                idx=i, m=m, n=n, n_nodes=len(adj), hole=hole, adj_local=adj,
                method="ILP", S=S_ilp, time_sec=t_ilp_outer,
                ilp_status=ilp_status, ilp_status_class=ilp_status_class, ilp_optimal=int(ilp_optimal),
                ilp_success=int(ilp_success), ilp_fallback=int(ilp_fallback), ilp_attempted=int(ilp_attempted),
                ilp_time_sec=ilp_time_sec, ilp_gap_ref=ilp_gap_ref,
            )
            method_sizes["ILP"].append(len(S_ilp))
            method_covs["ILP"].append(_coverage_local(adj, S_ilp))

            # ---- Greedy baseline ----
            t1 = time.perf_counter()
            S_gr = greedy_dominating_set(adj)
            t_gr = time.perf_counter() - t1
            _write_row(
                wr,
                idx=i, m=m, n=n, n_nodes=len(adj), hole=hole, adj_local=adj,
                method="Greedy", S=S_gr, time_sec=t_gr,
                ilp_status=ilp_status, ilp_status_class=ilp_status_class, ilp_optimal=int(ilp_optimal),
                ilp_success=int(ilp_success), ilp_fallback=int(ilp_fallback), ilp_attempted=int(ilp_attempted),
                ilp_time_sec=ilp_time_sec, ilp_gap_ref=ilp_gap_ref,
            )
            method_sizes["Greedy"].append(len(S_gr))
            method_covs["Greedy"].append(_coverage_local(adj, S_gr))

            # ---- GNN raw ----
            t2 = time.perf_counter()
            N = data.x.size(0)
            batch = torch.zeros(N, dtype=torch.long, device=GLOBAL_DEVICE)
            with torch.no_grad():
                logits, _ = model(data.x, data.edge_index, batch=batch)
                probs = torch.sigmoid(logits)
            scores = _probs_to_numpy_scores(probs)
            S_raw = set(gnn_raw_only(adj, scores, threshold=PROB_SOFTREPAIR_THRESHOLD))
            t_raw = time.perf_counter() - t2
            raw_cov = domination_coverage(adj, S_raw)
            _write_row(
                wr,
                idx=i, m=m, n=n, n_nodes=len(adj), hole=hole, adj_local=adj,
                method=f"{model_name}_raw", S=sorted(S_raw), time_sec=t_raw,
                raw_size=len(S_raw), added=0, removed=0,
                completion_rate=0.0, gnn_contribution=1.0 if len(S_raw) > 0 else 0.0,
                raw_coverage=raw_cov,
                ilp_status=ilp_status, ilp_status_class=ilp_status_class, ilp_optimal=int(ilp_optimal),
                ilp_success=int(ilp_success), ilp_fallback=int(ilp_fallback), ilp_attempted=int(ilp_attempted),
                ilp_time_sec=ilp_time_sec, ilp_gap_ref=ilp_gap_ref,
            )
            method_sizes[f"{model_name}_raw"].append(len(S_raw))
            method_covs[f"{model_name}_raw"].append(raw_cov)

            # ---- GNN + repair pipeline ----
            t3 = time.perf_counter()
            S_gnn = gnn_raw_then_complete(
                adj,
                probs,
                threshold=PROB_SOFTREPAIR_THRESHOLD,
                ilp_cutoff=ILP_CompleteTime,
                beta=PROB_SOFTREPAIR_BETA,
                verbose=PROB_SOFTREPAIR_VERBOSE,
            )
            t_gnn = time.perf_counter() - t3

            raw_size = len(S_raw)
            final_size = len(S_gnn)
            added = len(set(S_gnn) - S_raw)
            removed = len(S_raw - set(S_gnn))
            completion_rate = added / max(final_size, 1)
            gnn_contribution = len(set(S_gnn) & S_raw) / max(final_size, 1)
            raw_coverage = domination_coverage(adj, S_raw)

            repaired_name = f"{model_name}+PruneGuidedGreedyLocalSwap+ILPpolish"
            _write_row(
                wr,
                idx=i, m=m, n=n, n_nodes=len(adj), hole=hole, adj_local=adj,
                method=repaired_name,
                S=S_gnn, time_sec=t_gnn,
                raw_size=raw_size,
                added=added,
                removed=removed,
                completion_rate=completion_rate,
                gnn_contribution=gnn_contribution,
                raw_coverage=raw_coverage,
                ilp_status=ilp_status, ilp_status_class=ilp_status_class, ilp_optimal=int(ilp_optimal),
                ilp_success=int(ilp_success), ilp_fallback=int(ilp_fallback), ilp_attempted=int(ilp_attempted),
                ilp_time_sec=ilp_time_sec, ilp_gap_ref=ilp_gap_ref,
            )
            method_sizes[repaired_name].append(len(S_gnn))
            method_covs[repaired_name].append(_coverage_local(adj, S_gnn))

            pipeline_stats = getattr(gnn_raw_then_complete, "last_stats", {}) or {}
            repair_undom = len(get_undominated_vertices(adj, S_gnn))
            repair_valid = (repair_undom == 0)
            print(
                f"[4Methods-v.38] idx={i} | ILP={len(S_ilp)}({ilp_status_class}) Greedy={len(S_gr)} "
                f"Raw={len(S_raw)} Repair={len(S_gnn)} "
                f"valid={repair_valid} undom={repair_undom} | "
                f"added={added} removed={removed} completion={completion_rate:.3f} "
                f"contribution={gnn_contribution:.3f} raw_cov={raw_coverage:.3f}"
            )
            if PRINT_PIPELINE_LOG and pipeline_stats:
                stage_parts = [
                    f"{st['stage']}:{st['size']}/{st['cov']:.3f}/{st['undom']}"
                    for st in pipeline_stats.get("stages", [])
                    if st.get("stage") in {"raw", "prune", "adaptive-subgraph-ILP", "beam", "guided-greedy", "final"} or st.get("stage", "").startswith("micro-ILP#")
                ]
                print(
                    f"[Stages] idx={i} | local_only={pipeline_stats.get('local_only_mode', False)} "
                    f"heavy={pipeline_stats.get('heavy_repair_used', False)} | "
                    + " -> ".join(stage_parts[:12])
                )

            if (i + 1) % 10 == 0 or (i + 1) == len(graphs):
                print(f"[{i+1}/{len(graphs)}] done")

    def _avg(xs):
        return (sum(xs) / len(xs)) if xs else float("nan")

    repaired_name = f"{model_name}+PruneGuidedGreedyLocalSwap+ILPpolish"
    summary_rows = []
    for method, sizes in method_sizes.items():
        avg_size = _avg(sizes)
        avg_cov = _avg(method_covs.get(method, []))
        gap_vs_ilp_opt = ""
        gap_vs_ilp_lb = ""
        if method != "ILP":
            if ilp_opt_sizes:
                gap_vs_ilp_opt = avg_size - _avg(ilp_opt_sizes)
            elif ilp_all_sizes:
                gap_vs_ilp_lb = max(0.0, avg_size - _avg(ilp_all_sizes))
        summary_rows.append({
            "method": method,
            "avg_set_size": avg_size,
            "avg_coverage": avg_cov,
            "num_graphs": len(sizes),
            "gap_vs_ilp_opt": gap_vs_ilp_opt,
            "gap_vs_ilp_lb": gap_vs_ilp_lb,
            "ilp_opt_count": ilp_counts["optimal"],
            "ilp_feasible_nonoptimal_count": ilp_counts["feasible_nonoptimal"],
            "ilp_failed_or_missing_count": ilp_counts["failed_or_missing"],
        })

    with open(summary_csv_path, "w", newline="", encoding="utf-8") as fsum:
        fieldnames = [
            "method", "avg_set_size", "avg_coverage", "num_graphs",
            "gap_vs_ilp_opt", "gap_vs_ilp_lb",
            "ilp_opt_count", "ilp_feasible_nonoptimal_count", "ilp_failed_or_missing_count"
        ]
        wr_sum = csv.DictWriter(fsum, fieldnames=fieldnames)
        wr_sum.writeheader()
        wr_sum.writerows(summary_rows)

    avg_ilp_all = _avg(ilp_all_sizes)
    avg_ilp_opt = _avg(ilp_opt_sizes)
    avg_ilp_feas = _avg(ilp_feas_sizes)
    avg_repaired = _avg(method_sizes.get(repaired_name, []))
    if ilp_opt_sizes:
        gap_text = f"gap_vs_ilp_opt={avg_repaired - avg_ilp_opt:.4f}"
    elif ilp_all_sizes:
        gap_text = f"gap_vs_ilp_lb={max(0.0, avg_repaired - avg_ilp_all):.4f}"
    else:
        gap_text = "gap_vs_ilp=N/A"

    print(
        f"[v.38-summary] ILP avg(all)={avg_ilp_all:.4f} | "
        f"ILP avg(opt)={avg_ilp_opt:.4f} | ILP avg(feasible_nonopt)={avg_ilp_feas:.4f} | "
        f"opt={ilp_counts['optimal']} feasible_nonopt={ilp_counts['feasible_nonoptimal']} failed_or_missing={ilp_counts['failed_or_missing']} | "
        f"{gap_text}"
    )
    print(f"✔ TestSet 評估完成（4 methods, v.38），detail CSV -> {csv_path}")
    print(f"✔ TestSet 摘要完成（v.38），summary CSV -> {summary_csv_path}")


# =========================================================
#  7 個 GNN 模型：GCN / GATv2 / SAGE / GIN / TransformerConv / GraphTransformer / GPSConv
# =========================================================

class DomGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=GLOBAL_GNN_Layer):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch=None):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        logits = self.out(x).view(-1)
        if batch is None:
            graph_emb = x.mean(dim=0, keepdim=True)
        else:
            graph_emb = global_mean_pool(x, batch)
        return logits, graph_emb

class DomGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=GLOBAL_GNN_Layer):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch=None):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        logits = self.out(x).view(-1)
        if batch is None:
            graph_emb = x.mean(dim=0, keepdim=True)
        else:
            graph_emb = global_mean_pool(x, batch)
        return logits, graph_emb


class DomGATv2(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, heads=4, num_layers=GLOBAL_GNN_Layer):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(in_dim, hidden_dim // heads, heads=heads, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, concat=True))
        self.convs.append(GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, concat=True))
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch=None):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
        logits = self.out(x).view(-1)
        if batch is None:
            graph_emb = x.mean(dim=0, keepdim=True)
        else:
            graph_emb = global_mean_pool(x, batch)
        return logits, graph_emb


class DomSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=GLOBAL_GNN_Layer):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch=None):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        logits = self.out(x).view(-1)
        if batch is None:
            graph_emb = x.mean(dim=0, keepdim=True)
        else:
            graph_emb = global_mean_pool(x, batch)
        return logits, graph_emb


class DomGIN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=GLOBAL_GNN_Layer):
        super().__init__()
        self.convs = nn.ModuleList()
        nn1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs.append(GINConv(nn1))
        for _ in range(num_layers - 2):
            nnk = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(nnk))
        nn_last = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs.append(GINConv(nn_last))
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch=None):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        logits = self.out(x).view(-1)
        if batch is None:
            graph_emb = x.mean(dim=0, keepdim=True)
        else:
            graph_emb = global_mean_pool(x, batch)
        return logits, graph_emb


class DomTransformerConv(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, heads=4, num_layers=GLOBAL_GNN_Layer):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(TransformerConv(in_dim, hidden_dim // heads, heads=heads, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True))
        self.convs.append(TransformerConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True))
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch=None):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        logits = self.out(x).view(-1)
        if batch is None:
            graph_emb = x.mean(dim=0, keepdim=True)
        else:
            graph_emb = global_mean_pool(x, batch)
        return logits, graph_emb

class DomGraphTransformer(nn.Module):
    """
    簡化版 GraphTransformer：
    - 不做 padding / key_padding_mask
    - 把整個 batch 當一個長度為 N 的序列來跑 TransformerEncoder
    - 再用 global_mean_pool 按 batch 做圖層級 embedding
    """
    def __init__(self, in_dim, hidden_dim=64, num_layers=GLOBAL_GNN_Layer, heads=4):
        super().__init__()
        self.lin_in = nn.Linear(in_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,   # 輸入 shape 會是 (seq_len, batch, hidden)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch=None):
        # x: [N, in_dim]
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = self.lin_in(x)              # [N, hidden_dim]

        # 將整個 batch 的節點當成一條長序列，batch 維設成 1
        src = h.unsqueeze(1)            # [N, 1, hidden_dim]  (seq_len, batch, hidden)

        encoded = self.transformer_encoder(src)   # [N, 1, hidden_dim]
        h_enc = encoded.squeeze(1)      # [N, hidden_dim]

        logits = self.out(h_enc).view(-1)

        # 圖層級 embedding：依 batch 做 mean-pool
        graph_emb = global_mean_pool(h_enc, batch)
        return logits, graph_emb

"""
class DomGraphTransformer(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=4, heads=4):
        super().__init__()
        self.lin_in = nn.Linear(in_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True, # False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = self.lin_in(x)
        batch_size = int(batch.max().item() + 1)
        max_len = 0
        counts = []
        for b in range(batch_size):
            idx = (batch == b).nonzero(as_tuple=False).view(-1)
            counts.append(len(idx))
            if len(idx) > max_len:
                max_len = len(idx)

        padded = []
        attn_mask_list = []
        for b in range(batch_size):
            idx = (batch == b).nonzero(as_tuple=False).view(-1)
            cur = h[idx]
            pad_len = max_len - cur.size(0)
            if pad_len > 0:
                pad = torch.zeros(pad_len, h.size(1), device=h.device)
                cur = torch.cat([cur, pad], dim=0)
            padded.append(cur.unsqueeze(1))
            mask = torch.zeros(max_len, dtype=torch.bool, device=h.device)
            if counts[b] < max_len:
                mask[counts[b]:] = True
            attn_mask_list.append(mask.unsqueeze(0))

        padded = torch.cat(padded, dim=1)
        src_key_padding_mask = torch.cat(attn_mask_list, dim=0)
        encoded = self.transformer_encoder(padded, src_key_padding_mask=src_key_padding_mask)
        graph_embs = []
        for b in range(batch_size):
            length = counts[b]
            if length > 0:
                graph_embs.append(encoded[:length, b, :].mean(dim=0, keepdim=True))
            else:
                graph_embs.append(torch.zeros(1, h.size(1), device=h.device))
        graph_emb = torch.cat(graph_embs, dim=0)

        logits = self.out(h).view(-1)
        return logits, graph_emb
"""
  
class DomGPSConv(nn.Module):
    """GPS-style (Local + Global) model.

    Local path:  SAGEConv
    Global path: TransformerConv (heads, concat=False)
    FFN:         residual + LayerNorm

    Returns:
        logits    : shape (num_nodes,)  (node-level)
        graph_emb : pooled embedding    (graph-level)
    """

    def __init__(self, in_dim, hidden_dim=64, num_layers=GLOBAL_GNN_Layer, heads=4, dropout=0.10):
        super().__init__()
        self.node_encoder = nn.Linear(in_dim, hidden_dim)
        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.gconvs = nn.ModuleList([
            TransformerConv(hidden_dim, hidden_dim, heads=heads, concat=False)
            for _ in range(num_layers)
        ])

        self.norms1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])

        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = self.node_encoder(x)

        for i in range(self.num_layers):
            h_local = self.convs[i](h, edge_index)
            h_global = self.gconvs[i](h, edge_index)

            h = h + F.dropout(h_local + h_global, p=self.dropout, training=self.training)
            h = self.norms1[i](h)

            h_ffn = self.ffns[i](h)
            h = h + F.dropout(h_ffn, p=self.dropout, training=self.training)
            h = self.norms2[i](h)

        logits = self.out(h).view(-1)
        graph_emb = global_mean_pool(h, batch)
        return logits, graph_emb


"""
class DomGPSConv(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=GLOBAL_GNN_Layer, heads=4):
        super().__init__()
        self.node_encoder = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList()
        self.attn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GPSConv(
                local_nn=GCNConv(hidden_dim, hidden_dim),
                global_attn=GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, concat=True)
            ))
            self.attn_layers.append(nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=heads, batch_first=True
            ))
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = self.node_encoder(x)
        for conv, attn in zip(self.layers, self.attn_layers):
            h = conv(h, edge_index)
            h = F.relu(h)
            h_reshaped = h.unsqueeze(0)
            h_attn, _ = attn(h_reshaped, h_reshaped, h_reshaped)
            h = h + h_attn.squeeze(0)
            h = F.relu(h)
        logits = self.out(h).view(-1)
        graph_emb = global_mean_pool(h, batch)
        return logits, graph_emb
"""
        
############################################################
# RL fine-tuning: Sequential Actor–Critic for all GNN models
############################################################

# ...（此處以下保留你原本 v11_Experiments.py 後面所有內容：
#      模型定義、訓練流程、RL fine-tuning、實驗模式 run_experiment_multi_size_and_holes、
#      實驗結果視覺化、主選單 main_menu() 等）
#
# 由於整支程式非常長，我在這裡示意到關鍵修改區塊。
# 你可以將上述內容貼回你現有 v11_Experiments.py 的對應位置，
# 或直接用這份檔案覆蓋，再把後半段（模型＋實驗＋選單）從原檔貼到這份檔案下方。


############################################################
# RL fine-tuning: Sequential Actor–Critic for all GNN models
############################################################





# ===================== CLEAN RL INPUT-DIM HELPERS =====================
def _infer_model_in_dim(model):
    """
    Infer input feature dimension from live model instance.
    Supports DomGCN / DomGATv2 / DomSAGE / DomGIN / DomTransformerConv /
    DomGraphTransformer / DomGPSConv.
    """
    import torch.nn as nn

    model_type = type(model).__name__
    alias = {
        "DomGCN": "GCN",
        "DomGATv2": "GATv2",
        "DomSAGE": "SAGE",
        "DomGIN": "GIN",
        "DomTransformerConv": "TRANSFORMER",
        "DomGraphTransformer": "GRAPHORMER",
        "DomGPSConv": "GPSCONV",
    }
    model_name = alias.get(model_type, model_type)

    if model_name == "GRAPHORMER" and hasattr(model, "lin_in") and hasattr(model.lin_in, "weight"):
        return int(model.lin_in.weight.shape[1])

    if model_name == "GPSCONV":
        if hasattr(model, "node_encoder") and hasattr(model.node_encoder, "weight"):
            return int(model.node_encoder.weight.shape[1])
        if hasattr(model, "lin_in") and hasattr(model.lin_in, "weight"):
            return int(model.lin_in.weight.shape[1])

    if hasattr(model, "convs") and len(model.convs) > 0:
        first_conv = model.convs[0]

        if hasattr(first_conv, "lin") and hasattr(first_conv.lin, "weight"):
            return int(first_conv.lin.weight.shape[1])

        if hasattr(first_conv, "nn"):
            try:
                for layer in first_conv.nn:
                    if isinstance(layer, nn.Linear):
                        return int(layer.weight.shape[1])
            except TypeError:
                pass

        for attr in ("lin_l", "lin_r", "lin_key", "lin_query", "lin_value"):
            if hasattr(first_conv, attr):
                lin = getattr(first_conv, attr)
                if hasattr(lin, "weight"):
                    return int(lin.weight.shape[1])

    raise ValueError(f"Cannot infer model input dimension for model type {model_type}.")


def _resolve_model_in_dim(model):
    """
    Prefer the globally remembered training input dimension.
    Fall back to live model inspection only if needed.
    """
    global GLOBAL_MODEL_IN_DIM
    try:
        if GLOBAL_MODEL_IN_DIM is not None:
            return int(GLOBAL_MODEL_IN_DIM)
    except Exception:
        pass
    return int(_infer_model_in_dim(model))
# ==========================================================



# ===================== STOP-ACTION RL MODULE =====================
def _augment_logits_with_stop(logits, stop_logit):
    if logits.dim() == 1:
        return torch.cat([logits, stop_logit.view(1)], dim=0)
    return torch.cat([logits, stop_logit.view(1, 1)], dim=0)


def _compute_stop_logit_from_node_logits(node_logits):
    """
    Lightweight stop head without changing model architecture:
    use a function of current node logits as an implicit stop score.
    This keeps the patch architecture-compatible with all existing GNNs.
    """
    if node_logits.numel() == 0:
        return torch.tensor(0.0, device=node_logits.device, dtype=node_logits.dtype)
    # Conservative stop score: slightly below the mean initially,
    # but can dominate when valid node scores become small.
    return node_logits.mean() + 0.10


def _dominated_ratio(adj, S):
    N = len(adj)
    if N == 0:
        return 1.0
    dom = set(int(v) for v in S)
    for v in list(dom):
        for u in adj[v]:
            dom.add(int(u))
    return len(dom) / N


def rl_rollout_one_graph_with_stop(model, data, adj, device=None, max_steps=None,
                                   stop_min_coverage=RL_STOP_MIN_COVERAGE):
    """
    RL rollout with a STOP action.
    v35 focus: keep raw domination high, but shrink raw set more aggressively.
    Actions:
      0..N-1 : choose a vertex
      N      : STOP
    """
    if device is None:
        device = data.x.device

    model.train()
    N = len(adj)
    if N == 0:
        return [], [], [], [], 0.0, []

    if max_steps is None:
        max_steps = min(RL_MAX_STEPS_PER_GRAPH, N)
    else:
        max_steps = min(int(max_steps), RL_MAX_STEPS_PER_GRAPH, N)

    selected = set()
    dominated = set()

    log_probs = []
    entropies = []
    rewards = []
    actions = []

    batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

    for step in range(max_steps):
        logits, _ = _clean_model_forward_logits_and_graph_emb(model, data, batch)
        logits = logits.squeeze(-1)

        valid_node_mask = torch.zeros(N, device=logits.device, dtype=logits.dtype)
        marginal_gains = {}
        for v in range(N):
            if v in selected:
                continue
            gain = 1 + sum(1 for u in adj[v] if u not in dominated)
            if v not in dominated:
                valid_node_mask[v] = 1.0
                marginal_gains[v] = gain
            else:
                extra_gain = gain - 1
                if (not RL_ACTION_MASK_DOMINATED) and extra_gain > 0:
                    valid_node_mask[v] = 1.0
                    marginal_gains[v] = gain

        if float(valid_node_mask.sum().item()) <= 0.0:
            break

        masked_node_logits = logits.clone()
        masked_node_logits[valid_node_mask <= 0] = -1e9

        coverage_now = len(dominated) / max(N, 1)
        valid_logits = masked_node_logits[valid_node_mask > 0]
        best_valid_logit = valid_logits.max()
        stop_logit = best_valid_logit - 0.05
        if coverage_now < stop_min_coverage:
            stop_logit = stop_logit - 2.00
        elif coverage_now < RL_STOP_TARGET_COVERAGE:
            stop_logit = stop_logit - 0.75
        else:
            stop_logit = stop_logit + 0.15

        all_logits = torch.cat([masked_node_logits, stop_logit.view(1)], dim=0)
        dist = torch.distributions.Categorical(logits=all_logits)
        action = dist.sample()

        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())

        a = int(action.item())
        actions.append(a)

        if a == N:
            if coverage_now >= RL_STOP_TARGET_COVERAGE:
                stop_reward = RL_STOP_NEAR_DONE_BONUS - 0.10 * (len(selected) / max(N, 1))
            else:
                stop_reward = -(RL_STOP_TOO_EARLY_PENALTY + 3.0 * (RL_STOP_TARGET_COVERAGE - coverage_now))
            rewards.append(float(stop_reward))
            break

        prev_cov = len(dominated)

        selected.add(a)
        dominated.add(a)
        for u in adj[a]:
            dominated.add(int(u))

        new_cov = len(dominated)
        gain = new_cov - prev_cov

        remaining = N - new_cov
        future_penalty = remaining / max(N, 1)
        delta_coverage = gain / max(N, 1)
        current_size_ratio = len(selected) / max(N, 1)
        done = (new_cov == N)

        reward = (
            0.65 * RL_REWARD_GAIN_WEIGHT * float(gain)
            - max(RL_STEP_SIZE_PENALTY, 0.06)
            - RL_SHRINK_FUTURE_PENALTY_WEIGHT * future_penalty
            + RL_SHRINK_DELTA_COVERAGE_WEIGHT * delta_coverage
            - RL_SHRINK_CURRENT_SIZE_WEIGHT * current_size_ratio
        )

        if gain <= 1:
            reward -= max(RL_REWARD_REDUNDANCY_PENALTY, RL_SHRINK_LOW_GAIN_PENALTY)

        if done:
            reward += max(0.50, RL_TERMINAL_DOMINATION_BONUS * 0.70)
            reward -= RL_SHRINK_DONE_SIZE_WEIGHT * len(selected)

        rewards.append(float(reward))

        if done:
            break

    if rewards and len(dominated) < N:
        rewards[-1] -= max(RL_UNCOVERED_PENALTY, 2.0)
        rewards[-1] -= 6.0

    return log_probs, entropies, rewards, actions, float(sum(rewards)), sorted(selected)

def _rl_select_raw_set_with_stop(model, data, adj, device=None, max_steps=None, stop_min_coverage=RL_STOP_MIN_COVERAGE):
    if device is None:
        device = data.x.device
    model.eval()

    N = len(adj)
    if N == 0:
        return []

    if max_steps is None:
        max_steps = min(RL_MAX_STEPS_PER_GRAPH, N)
    else:
        max_steps = min(int(max_steps), RL_MAX_STEPS_PER_GRAPH, N)

    selected = set()
    dominated = set()
    batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

    with torch.no_grad():
        for step in range(max_steps):
            logits, _ = _clean_model_forward_logits_and_graph_emb(model, data, batch)
            logits = logits.squeeze(-1)

            valid = []
            for v in range(N):
                if v in selected:
                    continue
                if v not in dominated:
                    valid.append(v)
                else:
                    gain = sum(1 for u in adj[v] if u not in dominated)
                    if (not RL_ACTION_MASK_DOMINATED) and gain > 0:
                        valid.append(v)

            if not valid:
                break

            coverage_now = len(dominated) / max(N, 1)
            valid_logits = logits[valid]
            best_valid_logit = valid_logits.max()

            stop_logit = best_valid_logit - 0.05
            if coverage_now < stop_min_coverage:
                stop_logit = stop_logit - 2.00
            elif coverage_now < RL_STOP_TARGET_COVERAGE:
                stop_logit = stop_logit - 0.75
            else:
                stop_logit = stop_logit + 0.15

            best_v = None
            best_score = None
            best_gain = -1
            for v in valid:
                score = float(logits[v].item())
                gain = 1 + sum(1 for u in adj[v] if u not in dominated)
                if (best_score is None or score > best_score or
                    (abs(score - best_score) <= 1e-12 and gain > best_gain)):
                    best_score = score
                    best_gain = gain
                    best_v = v

            stop_score = float(stop_logit.item())
            if coverage_now >= RL_STOP_TARGET_COVERAGE and stop_score >= float(best_score) + 0.05:
                break

            selected.add(int(best_v))
            dominated.add(int(best_v))
            for u in adj[best_v]:
                dominated.add(int(u))

            if len(dominated) == N:
                break

    return sorted(selected)

# ==========================================================

def prune_dominating_set(adj, D):
    """
    Given a dominating set D, greedily remove redundant vertices
    while preserving domination.
    """
    D = set(int(v) for v in D)

    def is_dom(S):
        dominated = set(S)
        for v in S:
            for u in adj[v]:
                dominated.add(int(u))
        return len(dominated) == len(adj)

    changed = True
    while changed:
        changed = False
        for v in list(D):
            T = D - {v}
            if is_dom(T):
                D.remove(v)
                changed = True

    return sorted(D)

# ===================== CLEAN RL EVAL HELPERS =====================
def _is_dominating_set(adj, D):
    Dset = set(int(v) for v in D)
    N = len(adj)
    dominated = set(Dset)
    for v in Dset:
        for u in adj[v]:
            dominated.add(int(u))
    return len(dominated) == N


def _greedy_domination_set(adj):
    N = len(adj)
    undom = set(range(N))
    D = []
    while undom:
        best_v = None
        best_gain = -1
        for v in range(N):
            cover = {v} | set(int(u) for u in adj[v])
            gain = len(undom & cover)
            if gain > best_gain:
                best_gain = gain
                best_v = v
        if best_v is None:
            break
        D.append(best_v)
        undom.discard(best_v)
        for u in adj[best_v]:
            undom.discard(int(u))
    return D


def _rl_select_raw_set(model, data, adj, device=None, max_steps=None):
    if device is None:
        device = data.x.device
    model.eval()

    N = len(adj)
    if N == 0:
        return []

    if max_steps is None:
        max_steps = min(RL_MAX_STEPS_PER_GRAPH, N)
    else:
        max_steps = min(int(max_steps), RL_MAX_STEPS_PER_GRAPH, N)

    selected = set()
    dominated = set()
    batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

    with torch.no_grad():
        for step in range(max_steps):
            logits, _ = _clean_model_forward_logits_and_graph_emb(model, data, batch)
            logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits)

            valid = []
            for v in range(N):
                if v in selected:
                    continue
                if v not in dominated:
                    valid.append(v)
                else:
                    gain = sum(1 for u in adj[v] if u not in dominated)
                    if gain > 0:
                        valid.append(v)

            if not valid:
                break

            best_v = None
            best_score = None
            for v in valid:
                score = float(probs[v].item())
                if best_score is None or score > best_score:
                    best_score = score
                    best_v = v

            if best_v is None:
                break

            selected.add(int(best_v))
            dominated.add(int(best_v))
            for u in adj[best_v]:
                dominated.add(int(u))

            if len(dominated) == N:
                break

    return sorted(selected)


def _complete_domination_from_initial(adj, initial_set):
    D = set(int(v) for v in initial_set)
    N = len(adj)

    def dominated_vertices(S):
        dom = set(S)
        for v in S:
            for u in adj[v]:
                dom.add(int(u))
        return dom

    dominated = dominated_vertices(D)
    while len(dominated) < N:
        undom = set(range(N)) - dominated
        best_v = None
        best_gain = -1
        for v in range(N):
            if v in D:
                continue
            cover = {v} | set(int(u) for u in adj[v])
            gain = len(undom & cover)
            if gain > best_gain:
                best_gain = gain
                best_v = v
        if best_v is None:
            break
        D.add(int(best_v))
        dominated = dominated_vertices(D)
    return sorted(D)

def compute_domination_coverage(adj, S):
    """
    Return coverage ratio: |N[S]| / |V|
    """
    N = len(adj)
    if N == 0:
        return 1.0

    dominated = set(int(v) for v in S)
    for v in S:
        for u in adj[v]:
            dominated.add(int(u))

    return len(dominated) / N

def _evaluate_rl_vs_greedy_ilp(model, eval_graphs, device=None, model_type="MODEL", max_eval_graphs=20):
    if device is None:
        device = next(model.parameters()).device

    if not eval_graphs:
        print(f"[Eval-{model_type}] no graphs")
        return

    subset = eval_graphs[:min(len(eval_graphs), max_eval_graphs)]

    rl_raw_sizes = []
    rl_completed_sizes = []
    greedy_sizes = []
    ilp_sizes_all = []
    ilp_sizes_opt = []
    ilp_nonoptimal_sizes = []
    raw_dom_count = 0
    raw_cov_sum = 0.0
    completed_dom_count = 0
    ilp_optimal_count = 0
    ilp_feasible_nonoptimal_count = 0
    ilp_failed_or_missing_count = 0
    cnt = len(subset)

    for g in subset:
        adj = g["adj"]
        data = g["data"]
        data = data.to(device)

        model_in_dim = _resolve_model_in_dim(model)
        data.x = _pad_or_trunc_x(data.x, model_in_dim).to(device)

        rl_raw = _rl_select_raw_set_with_stop(
            model, data, adj, device=device,
            max_steps=min(len(adj), RL_MAX_STEPS_PER_GRAPH)
        )

        raw_cov = compute_domination_coverage(adj, rl_raw)
        raw_cov_sum += raw_cov

        rl_completed = _complete_domination_from_initial(adj, rl_raw)
        rl_completed = prune_dominating_set(adj, rl_completed)
        rl_completed = local_improve_dominating_set_fast(adj, rl_completed)
        rl_completed = prune_dominating_set(adj, rl_completed)

        greedy_set = _greedy_domination_set(adj)

        ilp_size = None
        try:
            ilp_size = int(float(g["labels_ilp"].sum().item()))
        except Exception:
            ilp_size = None

        ilp_status = g.get("ilp_status_class", g.get("ilp_status", "UNKNOWN"))
        ilp_status_class = _normalize_ilp_status(ilp_status)
        ilp_optimal = bool(g.get("ilp_optimal", False)) or (ilp_status_class == "OPTIMAL")
        ilp_feasible_nonoptimal = (ilp_status_class == "TIME_LIMIT_FEASIBLE") and (not ilp_optimal)

        rl_raw_sizes.append(len(rl_raw))
        rl_completed_sizes.append(len(rl_completed))
        greedy_sizes.append(len(greedy_set))
        if ilp_size is not None:
            ilp_sizes_all.append(ilp_size)
            if ilp_optimal:
                ilp_sizes_opt.append(ilp_size)
                ilp_optimal_count += 1
            elif ilp_feasible_nonoptimal:
                ilp_nonoptimal_sizes.append(ilp_size)
                ilp_feasible_nonoptimal_count += 1
            else:
                ilp_failed_or_missing_count += 1
        else:
            ilp_failed_or_missing_count += 1

        if _is_dominating_set(adj, rl_raw):
            raw_dom_count += 1
        if _is_dominating_set(adj, rl_completed):
            completed_dom_count += 1

    def _avg(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    avg_raw = _avg(rl_raw_sizes)
    raw_cov_avg = raw_cov_sum / max(cnt, 1)
    avg_completed = _avg(rl_completed_sizes)
    avg_greedy = _avg(greedy_sizes)
    avg_ilp_all = _avg(ilp_sizes_all)
    avg_ilp_opt = _avg(ilp_sizes_opt)
    avg_ilp_feasible = _avg(ilp_nonoptimal_sizes)

    gap_vs_greedy = avg_completed - avg_greedy if greedy_sizes else float("nan")
    gap_vs_ilp_report = float("nan")
    gap_label = "gap_vs_ilp_opt"
    gap_note = ""

    if ilp_sizes_opt:
        gap_vs_ilp_report = avg_completed - avg_ilp_opt
        gap_label = "gap_vs_ilp_opt"
        if gap_vs_ilp_report < 0:
            gap_note = " [CHECK: completed beat stored OPTIMAL ILP labels]"
    elif ilp_sizes_all:
        raw_gap_all = avg_completed - avg_ilp_all
        gap_vs_ilp_report = max(0.0, raw_gap_all)
        gap_label = "gap_vs_ilp_lb"
        gap_note = " [non-optimal ILP baseline present; negative gaps clamped to 0 as lower-bound comparison]"

    print(
        f"[Eval-{model_type}] graphs={cnt}  "
        f"raw_avg={avg_raw:.2f}  "
        f"completed_avg={avg_completed:.2f}  "
        f"greedy_avg={avg_greedy:.2f}  "
        f"ilp_avg={avg_ilp_all:.2f}  "
        f"ilp_opt_avg={avg_ilp_opt:.2f}  "
        f"gap_vs_greedy={gap_vs_greedy:.2f}  "
        f"{gap_label}={gap_vs_ilp_report:.2f}  "
        f"raw_dom={raw_dom_count}/{cnt}  "
        f"raw_cov={raw_cov_avg:.3f}  "
        f"completed_dom={completed_dom_count}/{cnt}"
        f"  ilp_opt={ilp_optimal_count}/{cnt}"
        f"  ilp_feas_nonopt={ilp_feasible_nonoptimal_count}/{cnt}"
        f"  ilp_fail_missing={ilp_failed_or_missing_count}/{cnt}"
        f"{gap_note}"
    )

    if ilp_sizes_all and not ilp_sizes_opt:
        print(
            f"[Eval-{model_type}] NOTE: No proven optimal ILP labels in eval subset; "
            f"reported {gap_label} uses time-limited ILP as a lower-bound style baseline. "
            f"avg_ilp_feasible={avg_ilp_feasible:.2f}"
        )
# ==========================================================

# ===================== RL / SAVE MISSING DEFINITIONS PATCH =====================
def safe_save_model_state_dict(model, weight_path):
    """
    Save model weights safely by moving tensors to CPU first when requested.
    """
    from pathlib import Path
    weight_path = Path(weight_path)
    weight_path.parent.mkdir(parents=True, exist_ok=True)

    state = model.state_dict()
    cpu_state = {}
    for k, v in state.items():
        try:
            cpu_state[k] = v.detach().cpu()
        except Exception:
            cpu_state[k] = v
    torch.save(cpu_state, str(weight_path))

import random

def local_improve_dominating_set_fast(adj, D, max_trials=50):
    D = set(int(v) for v in D)
    N = len(adj)

    def is_dom(S):
        dominated = set(S)
        for v in S:
            for u in adj[v]:
                dominated.add(int(u))
        return len(dominated) == N

    outside = [v for v in range(N) if v not in D]

    # 1-for-1 (sampled)
    for _ in range(max_trials):
        if not D or not outside:
            break
        old_v = random.choice(list(D))
        new_v = random.choice(outside)

        cand = (D - {old_v}) | {new_v}
        if is_dom(cand):
            return sorted(cand)

    # 2-for-1 (very limited)
    for _ in range(max_trials // 2):
        if len(D) < 2 or not outside:
            break
        old_u, old_v = random.sample(list(D), 2)
        new_v = random.choice(outside)

        cand = (D - {old_u, old_v}) | {new_v}
        if is_dom(cand):
            return sorted(cand)

    return sorted(D)

def train_actor_critic_for_model(model_type, model, graphs, adj_list,
                                 episodes=RL_FINE_EPOSODES,
                                 beta_size=RL_BETA,
                                 lambda_sup=0.10,
                                 entropy_beta=RL_ENTROPY,
                                 device=None):
    """
    Clean RL training with STOP action + every-20 evaluation.
    Final RL objective aligns with completed domination set size.
    """
    if device is None:
        device = next(model.parameters()).device

    if not graphs:
        return

    if RL_VALIDATE_GRAPHS:
        if RL_SKIP_INVALID_GRAPHS:
            graphs, skipped = _filter_valid_graphs_for_rl(
                graphs, device=device, verbose=True, model_type=model_type
            )
            if not graphs:
                print(f"[RL-{model_type}] 無可用 graph，略過 RL fine-tuning")
                return
        else:
            for i, g in enumerate(graphs):
                _validate_graph_for_rl(g, device=device)

    max_nodes_for_rl = (
        RL_MAX_GRAPH_N_FOR_RL_GPSCONV
        if str(model_type).upper() == "GPSCONV"
        else RL_MAX_GRAPH_N_FOR_RL
    )

    usable_pairs = [(g, adj) for g, adj in zip(graphs, adj_list) if len(adj) <= max_nodes_for_rl]
    if not usable_pairs:
        print(f"[RL-{model_type}] 無符合大小限制的 graph（max_nodes={max_nodes_for_rl}），跳過 RL fine-tuning")
        return

    graphs = [ga[0] for ga in usable_pairs]
    adj_list = [ga[1] for ga in usable_pairs]

    opt = torch.optim.Adam(model.parameters(), lr=RL_FINE_TUNE_LR)
    stabilizer = RLTrainingStabilizer(
        momentum=RL_BASELINE_MOMENTUM,
        eps=RL_REWARD_NORM_EPS,
        clamp=RL_ADVANTAGE_CLAMP,
    )

    order = sorted(range(len(graphs)), key=lambda i: graphs[i]["data"].x.size(0))

    for ep in range(1, episodes + 1):
        if ep < max(2, episodes // 2):
            used_indices = order[:max(1, len(order) // 2)]
        else:
            used_indices = order

        entropy_weight = max(float(entropy_beta) * (RL_ENTROPY_DECAY ** (ep - 1)), RL_ENTROPY_MIN)

        total_loss = 0.0
        total_policy = 0.0
        total_sup = 0.0
        total_entropy = 0.0
        total_reward = 0.0
        total_steps = 0.0
        total_completed = 0.0
        num_used = 0

        for idx in used_indices:
            g = graphs[idx]
            adj = adj_list[idx]

            data = apply_symmetry_augmentation_to_data(g["data"]).to(device)
            model_in_dim = _resolve_model_in_dim(model)
            data.x = _pad_or_trunc_x(data.x, model_in_dim).to(device)

            opt.zero_grad(set_to_none=True)

            log_probs, entropies, step_rewards, actions, shaping_sum, raw_set = rl_rollout_one_graph_with_stop(
                model, data, adj,
                device=device,
                max_steps=min(len(adj), RL_MAX_STEPS_PER_GRAPH),
                stop_min_coverage=0.5,
            )

            if not log_probs:
                continue

            completed_set = _complete_domination_from_initial(adj, raw_set)
            completed_set = prune_dominating_set(adj, completed_set)
            completed_set = local_improve_dominating_set_fast(adj, completed_set)
            completed_set = prune_dominating_set(adj, completed_set)
            
            completed_size = len(completed_set)
            raw_dom_ratio = _dominated_ratio(adj, raw_set)

            N = max(len(adj), 1)
            final_reward = (
                -RL_FINAL_COMPLETED_SIZE_WEIGHT * (float(completed_size) / N)
                -RL_FINAL_RAW_SIZE_WEIGHT * (len(raw_set) / N)
                +RL_FINAL_RAW_DOM_WEIGHT * float(raw_dom_ratio)
            )
            G = float(shaping_sum) + final_reward

            """
            stabilizer.update(G)
            adv_scalar = stabilizer.advantage(G)

            log_probs_t = torch.stack(log_probs)
            entropies_t = torch.stack(entropies)

            policy_loss = -float(adv_scalar) * log_probs_t.sum()
            entropy_term = entropies_t.mean()
            rl_loss = policy_loss - entropy_weight * entropy_term
            """
            adv_scalar = stabilizer.advantage(G)
            stabilizer.update(G)

            log_probs_t = torch.stack(log_probs)
            entropies_t = torch.stack(entropies)

            adv = torch.as_tensor(adv_scalar, dtype=log_probs_t.dtype, device=log_probs_t.device)
            adv = adv.clamp(-5.0, 5.0)

            policy_loss = -(adv * log_probs_t.mean())
            entropy_term = entropies_t.mean()
            rl_loss = policy_loss - entropy_weight * entropy_term

            """
            print(
                f"G={G:.4f}  adv={adv.item():.4f}  "
                f"steps={log_probs_t.numel()}  "
                f"logp_mean={log_probs_t.mean().item():.4f}  "
                f"policy={policy_loss.item():.4f}  "
                f"entropy={entropy_term.item():.4f}"
            )
            """


            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
            logits_sup, _ = _clean_model_forward_logits_and_graph_emb(model, data, batch)
            logits_sup = logits_sup.squeeze(-1)

            y_ilp = g["labels_ilp"].float().to(device)
            y_gr = g["labels_greedy"].float().to(device)
            ilp_w, greedy_w = get_dynamic_teacher_weights(g["data"].x.size(0))
            w_sum = ilp_w + greedy_w
            y_mix = (ilp_w * y_ilp + greedy_w * y_gr) / max(float(w_sum), 1e-12)
            sup_loss = F.binary_cross_entropy_with_logits(logits_sup, y_mix)

            loss = rl_loss + float(lambda_sup) * sup_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            if getattr(device, "type", "cpu") == "cuda":
                torch.cuda.empty_cache()

            total_loss += float(loss.detach().item())
            total_policy += float(policy_loss.detach().item())
            total_sup += float(sup_loss.detach().item())
            total_entropy += float(entropy_term.detach().item())
            total_reward += float(G)
            total_steps += float(len(log_probs))
            total_completed += float(completed_size)
            num_used += 1

        if ep == 1 or ep % 20 == 0 or ep == episodes:
            denom = max(num_used, 1)
            print(
                f"[RL-{model_type}] Episode {ep}/{episodes} "
                f"loss={total_loss/denom:.4f}  "
                f"policy={total_policy/denom:.4f}  "
                f"sup={total_sup/denom:.4f}  "
                f"entropy={total_entropy/denom:.4f}  "
                f"ent_w={entropy_weight:.5f}  "
                f"reward={total_reward/denom:.4f}  "
                f"avg_steps={total_steps/denom:.2f}  "
                f"completed_avg={total_completed/denom:.2f}"
            )
            try:
                _evaluate_rl_vs_greedy_ilp(
                    model,
                    graphs,
                    device=device,
                    model_type=model_type,
                    max_eval_graphs=20
                )
            except Exception as e:
                print(f"[Eval-{model_type}] 評估失敗：{e}")
# ==========================================================


# ===================== RL HELPER COMPATIBILITY =====================
def _reset_cuda_after_rl_error():
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _get_rl_graph_node_limit(model_type=None):
    if model_type is not None and str(model_type).upper() == "GPSCONV":
        return RL_MAX_GRAPH_N_FOR_RL_GPSCONV
    return RL_MAX_GRAPH_N_FOR_RL


def _validate_graph_for_rl(graph, device=None):
    data = graph["data"]
    N = int(data.x.size(0))
    if N <= 0:
        raise RuntimeError("[RL] empty graph (N<=0)")
    if data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
        raise RuntimeError(f"[RL] edge_index shape invalid: {tuple(data.edge_index.size())}")
    if data.edge_index.numel() > 0:
        ei_min = int(data.edge_index.min().item())
        ei_max = int(data.edge_index.max().item())
        if ei_min < 0 or ei_max >= N:
            raise RuntimeError(f"[RL] edge_index out of range: min={ei_min}, max={ei_max}, N={N}")

    adj = graph["adj"]
    if len(adj) != N:
        raise RuntimeError(f"[RL] len(adj)={len(adj)} != N={N}")
    for v in range(N):
        for u in adj[v]:
            if int(u) < 0 or int(u) >= N:
                raise RuntimeError(f"[RL] adj[{v}] contains out-of-range node {u} (N={N})")

    for key in ("labels_ilp", "labels_greedy"):
        y = graph.get(key, None)
        if y is None:
            raise RuntimeError(f"[RL] missing {key}")
        if int(y.numel()) != N:
            raise RuntimeError(f"[RL] {key}.numel()={int(y.numel())} != N={N}")
    return True


def _filter_valid_graphs_for_rl(graphs, device=None, verbose=True, model_type=None):
    valid_graphs = []
    skipped = []
    node_limit = _get_rl_graph_node_limit(model_type)
    for idx, g in enumerate(graphs):
        try:
            _validate_graph_for_rl(g, device=device)
            N = int(g["data"].x.size(0))
            if node_limit is not None and N > int(node_limit):
                raise RuntimeError(f"graph too large for RL: N={N} > RL node limit={node_limit}")
            valid_graphs.append(g)
        except Exception as e:
            skipped.append((idx, str(e)))
    if verbose:
        label = f"-{model_type}" if model_type else ""
        print(f"[RL{label}] valid graphs for RL = {len(valid_graphs)} / {len(graphs)}")
        if skipped:
            max_show = min(8, len(skipped))
            for idx, msg in skipped[:max_show]:
                print(f"[RL{label}] skip graph #{idx}: {msg}")
            if len(skipped) > max_show:
                print(f"[RL{label}] ... and {len(skipped)-max_show} more skipped graphs")
    return valid_graphs, skipped
# ==========================================================

# ===================== CLEAN SINGLE RL MODULE =====================
class RLTrainingStabilizer:
    """
    Minimal, self-contained stabilizer used by the clean RL pipeline only.
    It intentionally does NOT depend on the old v23/v24 helper API.
    """
    def __init__(self, momentum=0.9, eps=1e-8, clamp=5.0):
        self.baseline = None
        self.var = 0.0
        self.momentum = momentum
        self.eps = eps
        self.clamp = clamp

    def update(self, reward):
        r = float(reward)
        if self.baseline is None:
            self.baseline = r
            self.var = 0.0
        else:
            delta = r - self.baseline
            self.baseline = self.momentum * self.baseline + (1.0 - self.momentum) * r
            self.var = self.momentum * self.var + (1.0 - self.momentum) * (delta ** 2)

    def advantage(self, reward):
        r = float(reward)
        if self.baseline is None:
            return r
        std = max((self.var + self.eps) ** 0.5, 1.0)
        adv = (r - self.baseline) / std
        return max(min(adv, self.clamp), -self.clamp)


def _clean_model_forward_logits_and_graph_emb(model, data, batch):
    """
    Robust forward wrapper:
    - preferred: model(x, edge_index, batch=batch) -> (node_logits, graph_emb)
    - fallback : model(x, edge_index) -> node_logits
    Graph embedding is optional for the clean RL path.
    """
    out = None
    try:
        out = model(data.x, data.edge_index, batch=batch)
    except TypeError:
        out = model(data.x, data.edge_index)

    if isinstance(out, tuple):
        if len(out) >= 2:
            return out[0], out[1]
        if len(out) == 1:
            return out[0], None
    return out, None


def rl_rollout_one_graph(model, data, adj, device=None, max_steps=None):
    """
    Clean memory-safe rollout for domination.
    Returns tensors ready for REINFORCE-style updates.
    """
    if device is None:
        device = data.x.device

    model.train()

    N = len(adj)
    if N == 0:
        return [], [], [], 0.0

    if max_steps is None:
        max_steps = min(RL_MAX_STEPS_PER_GRAPH, N)
    else:
        max_steps = min(int(max_steps), RL_MAX_STEPS_PER_GRAPH, N)

    selected = set()
    dominated = set()

    log_probs = []
    entropies = []
    rewards = []

    batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

    for step in range(max_steps):
        logits, _ = _clean_model_forward_logits_and_graph_emb(model, data, batch)
        logits = logits.squeeze(-1)
        probs = torch.sigmoid(logits)

        valid_mask = torch.zeros(N, device=probs.device, dtype=probs.dtype)
        for v in range(N):
            if v in selected:
                continue
            if v not in dominated:
                valid_mask[v] = 1.0
            else:
                gain = sum(1 for u in adj[v] if u not in dominated)
                if (not RL_ACTION_MASK_DOMINATED) and gain > 0:
                    valid_mask[v] = 1.0

        probs = probs * valid_mask
        total_prob = probs.sum()

        if not torch.isfinite(total_prob) or float(total_prob.item()) <= 0.0:
            candidates = [v for v in range(N) if (v not in selected and (v not in dominated or not RL_ACTION_MASK_DOMINATED))]
            if not candidates:
                break
            action_idx = max(candidates, key=lambda x: 1 + sum(1 for u in adj[x] if u not in dominated))
            p = torch.zeros(N, device=probs.device, dtype=probs.dtype)
            p[action_idx] = 1.0
            dist = torch.distributions.Categorical(p)
            action = torch.tensor(action_idx, device=probs.device)
        else:
            probs = probs / (total_prob + 1e-8)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())

        v = int(action.item())
        prev_cov = len(dominated)

        selected.add(v)
        dominated.add(v)
        for u in adj[v]:
            dominated.add(int(u))

        new_cov = len(dominated)
        gain = new_cov - prev_cov

        remaining = N - new_cov
        future_penalty = remaining / max(N, 1)
        delta_coverage = gain / max(N, 1)
        done = (new_cov == N)

        current_size_ratio = len(selected) / max(N, 1)
        reward = (
            0.65 * RL_REWARD_GAIN_WEIGHT * float(gain)
            - max(RL_STEP_SIZE_PENALTY, 0.06)
            - RL_SHRINK_FUTURE_PENALTY_WEIGHT * future_penalty
            + RL_SHRINK_DELTA_COVERAGE_WEIGHT * delta_coverage
            - RL_SHRINK_CURRENT_SIZE_WEIGHT * current_size_ratio
        )
        if gain <= 1:
            reward -= max(RL_REWARD_REDUNDANCY_PENALTY, RL_SHRINK_LOW_GAIN_PENALTY)
        if done:
            reward += max(0.50, RL_TERMINAL_DOMINATION_BONUS * 0.70)
            reward -= RL_SHRINK_DONE_SIZE_WEIGHT * len(selected)

        rewards.append(float(reward))

        if done:
            break

    if rewards and len(dominated) < N:
        rewards[-1] -= RL_UNCOVERED_PENALTY
        rewards[-1] -= 5.0

    return log_probs, entropies, rewards, float(sum(rewards))



# =========================================================
#  功能 2：讀取訓練資料並訓練 (一次訓練 1 模型)
# =========================================================
def train_one_model_from_safe_folder():
    """
    (功能 5) 讀取 Train_<topology>_* 訓練集資料，並只訓練「一個」你選擇的 GNN（預設 GCN）。
    - 目標符合你的新流程：選擇某一個 GNN 來跑
    """
    global GLOBAL_MODELS, GLOBAL_DEVICE, GLOBAL_GRAPHS, SELECTED_GNN, GLOBAL_MODEL_IN_DIM

    # 0) 選擇要跑的 GNN
    choose_one_gnn()
    model_name = SELECTED_GNN

    # 1) 選擇資料夾
    folder = select_train_folder_gui()
    if not folder:
        print("❌ 未選擇資料夾，取消訓練。")
        return
    
    # 2) 載入資料
    graphs = load_training_graphs_safe(folder)
    if not graphs:
        print("❌ 找不到圖資料")
        return
    GLOBAL_GRAPHS = graphs

    # 3) 裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GLOBAL_DEVICE = device
    print(f"[Train] 使用 device = {device}")

    # 4) 對齊 feature 維度
    feat_dims = [g["data"].x.size(1) for g in graphs]
    max_dim = max(feat_dims)
    print(f"[Train] 原始 feature 維度集合：{sorted(set(feat_dims))} → 使用 padded_dim={max_dim}")
    for g in graphs:
        x_static = pad_features_to_dim(g["data"].x, max_dim)
        g["data"].x = _append_zero_state_features(x_static, extra_dim=RL_STATE_EXTRA_DIM)

    in_dim = max_dim + RL_STATE_EXTRA_DIM
    GLOBAL_MODEL_IN_DIM = in_dim
    print(f"[Train] RL state feature dim = {RL_STATE_EXTRA_DIM} ({', '.join(RL_STATE_FEATURE_NAMES)}) → model_in_dim={in_dim}")
    hidden_dim = GPSCONV_HIDDEN_DIM if model_name == "GPSCONV" else DEFAULT_HIDDEN_DIM

    # 5) 只建立「一個」模型
    def _build_model(name: str):
        if name == "GCN":
            return DomGCN(in_dim, hidden_dim=hidden_dim).to(device)
        elif name == "GATv2":
            return DomGATv2(in_dim, hidden_dim=hidden_dim).to(device)
        elif name == "SAGE":
            return DomSAGE(in_dim, hidden_dim=hidden_dim).to(device)
        elif name == "GIN":
            return DomGIN(in_dim, hidden_dim=hidden_dim).to(device)
        elif name == "TRANSFORMER":
            return DomTransformerConv(in_dim, hidden_dim=hidden_dim).to(device)
        elif name == "GRAPHORMER":
            return DomGraphTransformer(in_dim, hidden_dim=hidden_dim).to(device)
        elif name == "GPSCONV":
            return DomGPSConv(in_dim, hidden_dim=hidden_dim, num_layers=GPSCONV_GNN_LAYERS, heads=GPSCONV_HEADS).to(device)
        else:
            raise ValueError(f"Unknown model: {name}")

    model = _build_model(model_name)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # 6) Supervised 訓練
    epochs = int(input("訓練 epoch 數 (例如 50): ").strip() or "50")
    print(f"[Train] Dual-teacher mixing: small graph (<{SMALL_GRAPH_NODE_THRESHOLD}) => ILP={SMALL_GRAPH_ILP_WEIGHT:.1f}, Greedy={SMALL_GRAPH_GREEDY_WEIGHT:.1f}; otherwise ILP={ILP_WEIGHT:.1f}, Greedy={GREEDY_WEIGHT:.1f}")
    gamma_imitation = 0.5

    print(f"[Train] 開始訓練：{model_name} ...")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting supervised -{timestamp}")

    for epoch in range(1, epochs + 1):
        #print(f"[Epoch {epoch}/{epochs}] {model_name}")
        random.shuffle(graphs)
        epoch_loss = 0.0
        for g in graphs:
            data = apply_symmetry_augmentation_to_data(g["data"]).to(device)
            data.x = _pad_or_trunc_x(data.x, in_dim).to(device)
            y_ilp = g["labels_ilp"].to(device)
            y_gr = g["labels_greedy"].to(device)
            ilp_w, greedy_w = get_dynamic_teacher_weights(g["data"].x.size(0))
            w_sum = ilp_w + greedy_w
            y_mix = (ilp_w * y_ilp + greedy_w * y_gr) / max(float(w_sum), 1e-12)

            N = data.x.size(0)
            batch = torch.zeros(N, dtype=torch.long, device=device)

            model.train()
            opt.zero_grad(set_to_none=True)
            logits, _ = model(data.x, data.edge_index, batch=batch)
            probs = torch.sigmoid(logits)

            bce_sup = F.binary_cross_entropy(probs, y_mix)

            # imitation (KL) terms
            eps = 1e-6
            p_mix = y_mix
            kl_ilp = (y_ilp * (torch.log(y_ilp + eps) - torch.log(p_mix + eps))).mean()
            kl_gr  = (y_gr  * (torch.log(y_gr  + eps) - torch.log(p_mix + eps))).mean()
            imitation = kl_ilp + kl_gr

            # domination coverage proxy loss
            cov = compute_coverage(g["adj"], gnn_raw_then_complete(g["adj"], probs, threshold=PROB_SOFTREPAIR_THRESHOLD, ilp_cutoff=ILP_CompleteTime, beta=PROB_SOFTREPAIR_BETA, verbose=PROB_SOFTREPAIR_VERBOSE))
            cov_t = torch.tensor(cov, dtype=torch.float32, device=probs.device).clamp(min=1e-6)
            dom_loss = -torch.log(cov_t)

            loss = bce_sup + gamma_imitation * imitation + 0.1 * dom_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()

            epoch_loss += float(loss.item())
        
        avg_loss = epoch_loss / max(len(graphs), 1)
        #if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
        if epoch % 5 == 0 or epoch == epochs: 
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"{epoch} -{timestamp}")
        print(f"[Epoch {epoch}/{epochs}] {model_name} loss={avg_loss:.4f}")

    # 7) 可選：RL fine-tuning（只對選定模型）
    #do_rl = input("是否進行 RL fine-tuning（Actor-Critic）？(y/n) [預設 n]: ").strip().lower()
    #if do_rl == "y":
    
    print(f"\n=== RL fine-tuning ({model_name}) [v38] ===")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting RL -{timestamp}")
    
    rl_failed = False
    if USE_RL_FINE_TUNE:
        try:
            graphs_for_rl = graphs
            if RL_VALIDATE_GRAPHS and RL_SKIP_INVALID_GRAPHS:
                graphs_for_rl, skipped = _filter_valid_graphs_for_rl(graphs, device=device, verbose=True, model_type=model_name)
            elif RL_VALIDATE_GRAPHS:
                for idx, g in enumerate(graphs):
                    _validate_graph_for_rl(g, device=device)
                graphs_for_rl = graphs
                print(f"[RL-{model_name}] graph validation passed: {len(graphs_for_rl)} graphs")

            if not graphs_for_rl:
                print(f"[RL-{model_name}] 無有效 graph，跳過 RL fine-tuning")
            else:
                adj_list = [g['adj'] for g in graphs_for_rl]
                train_actor_critic_for_model(
                    model_name, model, graphs_for_rl, adj_list,
                    episodes=RL_FINE_EPOSODES,
                    beta_size=RL_BETA,
                    lambda_sup=RL_LAMBDA,
                    entropy_beta=RL_ENTROPY,
                    device=device,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
        except Exception as e:
            rl_failed = True
            print(f"[RL-{model_name}] 發生錯誤，略過 RL: {e}")
            _reset_cuda_after_rl_error()
    else:
        print(f"[RL-{model_name}] USE_RL_FINE_TUNE=False，跳過 RL fine-tuning")

    # 8) 儲存權重（只存一個模型）
    # 8) 儲存權重（檔名必須等於該 GNN 名稱）
    #ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    print(f"Ending RL -{timestamp}")
    
    folder_name = Path(folder).name   # 只取最後一層名稱
    MODELPATH.mkdir(parents=True, exist_ok=True)  # 確保資料夾存在
    weight_path = MODELPATH / f"{model_name}-{folder_name}-v38-{ts}.pt"
    try:
        if RL_SAFE_SAVE_TO_CPU:
            safe_save_model_state_dict(model, weight_path)
        else:
            torch.save(model.state_dict(), str(weight_path))
        print(f"[Train] ✔ 已儲存模型權重：{weight_path} (檔名={model_name}-{folder_name}-{ts}.pt)")
    except Exception as e:
        print(f"[Save-{model_name}] 儲存模型失敗：{e}")
        raise
    GLOBAL_MODELS = {model_name: model}
    print("[Train] 完成：GLOBAL_MODELS 目前包含", list(GLOBAL_MODELS.keys()))

def load_models_from_model_folder():
    """(功能 6) 從 2Models/ 選取一個模型權重並載入。
    流程：
    1) 從 MODELPATH (= 2Models/) 選取 *.pt。
    2) 由檔名前綴推回模型名稱（例如：GCN-train-20260305-120000.pt → GCN）。
    3) 由 checkpoint 推回 in_dim，並用 GLOBAL_GRAPHS（若尚未載入則要求選 Train_*）
       對齊 feature_dim（pad / trunc）。
    """
    global GLOBAL_MODELS, GLOBAL_DEVICE, GLOBAL_GRAPHS, SELECTED_GNN, GLOBAL_MODEL_IN_DIM

    # 1) 選擇 2Models/*.pt
    weight_path = select_model_file_gui()
    if not weight_path:
        print("❌ 未選擇模型檔，取消載入。")
        return
    print(f"[Load] 選擇模型檔：{weight_path}")

    # 2) device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GLOBAL_DEVICE = device
    print(f"[Load] 使用 device = {device}")

    # 3) 先讀 checkpoint（用來推回 in_dim）
    try:
        state = torch.load(str(weight_path), map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(str(weight_path), map_location=device)
    except Exception:
        state = torch.load(str(weight_path), map_location='cpu')

    # 4) 推回模型名稱（統一成 AVAILABLE_GNNS 的 spelling）
    raw_prefix = weight_path.stem.split("-")[0]
    model_name = normalize_gnn_name(raw_prefix)
    SELECTED_GNN = model_name
    print(f"[Load] 偵測模型類型：{model_name}")

    # 5) 從 checkpoint 推回模型輸入維度（不需要先載入 Train_*）
    in_dim = infer_in_dim_from_state_dict(model_name, state)
    hidden_dim = GPSCONV_HIDDEN_DIM if model_name == "GPSCONV" else DEFAULT_HIDDEN_DIM

    # 5b) 記錄下來，之後跑 TestSet 會用到
    global GLOBAL_MODEL_IN_DIM
    GLOBAL_MODEL_IN_DIM = in_dim

    # 5c) 若目前已經有 GLOBAL_GRAPHS（例如你先載入過 TrainSafe graphs），就順便對齊其 feature_dim（pad / trunc）
    graphs = GLOBAL_GRAPHS
    if graphs:
        for g in graphs:
            x = g["data"].x
            if x.size(1) < in_dim:
                g["data"].x = pad_features_to_dim(x, in_dim)
            elif x.size(1) > in_dim:
                g["data"].x = x[:, :in_dim]

    # 8) 建立模型
    def _build_model(name: str):
        if name == "GCN":
            return DomGCN(in_dim, hidden_dim=hidden_dim).to(device)
        elif name == "GATv2":
            return DomGATv2(in_dim, hidden_dim=hidden_dim).to(device)
        elif name == "SAGE":
            return DomSAGE(in_dim, hidden_dim=hidden_dim).to(device)
        elif name == "GIN":
            return DomGIN(in_dim, hidden_dim=hidden_dim).to(device)
        elif name == "TRANSFORMER":
            return DomTransformerConv(in_dim, hidden_dim=hidden_dim).to(device)
        elif name == "GRAPHORMER":
            return DomGraphTransformer(in_dim, hidden_dim=hidden_dim).to(device)
        elif name == "GPSCONV":
            return DomGPSConv(in_dim, hidden_dim=hidden_dim, num_layers=GPSCONV_GNN_LAYERS, heads=GPSCONV_HEADS).to(device)
        else:
            raise ValueError(f"Unknown model: {name}")

    model = _build_model(model_name)

    # 9) 載入權重
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        print(f"[Load][Warn] state_dict mismatch for {model_name}: {e}")
        print("[Load][Warn] Fallback to strict=False (missing params will be randomly initialized).")
        model.load_state_dict(state, strict=False)

    GLOBAL_MODELS = {model_name: model}
    print(f"[Load] ✔ 已載入模型：{model_name}")
    print(f"[Load] 權重檔：{weight_path}")


def test_current_model():
    global GLOBAL_MODELS, GLOBAL_DEVICE
    if not GLOBAL_MODELS or GLOBAL_DEVICE is None:
        print("❌ 尚未訓練模型")
        return

    print("\n=== 測試單一模型（Irregular 測試圖 + GuidedGreedy）===")
    print("可用模型：", list(GLOBAL_MODELS.keys()))
    model_name = list(GLOBAL_MODELS.keys())[0]
    print(f"[Test] 使用模型：{model_name}")
    if model_name not in GLOBAL_MODELS:
        print("❌ 該模型尚未訓練")
        return

    m = input_int_with_default("m", 20, min_value=1)
    n = input_int_with_default("n", 20, min_value=1)
    hole_ratio = input_float_with_default("不規則挖洞比例 hole_ratio (例如 0.2，直接 Enter = 0)", 0.0, min_value=0.0, max_value=0.95)
    global GLOBAL_M, GLOBAL_N, GLOBAL_HOLE
    GLOBAL_M = m
    GLOBAL_N = n
    GLOBAL_HOLE = hole_ratio

    adj, coords, hole_ratio = build_irregular_grid_adj(m, n, hole_ratio=hole_ratio, ensure_connected=True)
    data = build_full_features(m, n, adj, coords=coords, pe_dim=8, rwe_dim=16).to(GLOBAL_DEVICE)

    model = GLOBAL_MODELS[model_name]
    model.eval()
    model_in_dim = GLOBAL_MODEL_IN_DIM or _infer_model_in_dim(model)
    data.x = _pad_or_trunc_x(data.x, model_in_dim).to(GLOBAL_DEVICE)

    N = data.x.size(0)
    batch = torch.zeros(N, dtype=torch.long, device=GLOBAL_DEVICE)
    with torch.no_grad():
        logits, g_emb = model(data.x, data.edge_index, batch=batch)
        probs = torch.sigmoid(logits)

    S_gg = gnn_raw_then_complete(adj, probs, threshold=0.5, ilp_cutoff=ILP_CompleteTime, completion_mode=GNN_COMPLETION_MODE)
    S_gr = greedy_dominating_set(adj)
    S_ilp = ilp_minimum_dominating_set(adj)

    print("\n=== 比較 ===")
    print(f"ILP: |D|={len(S_ilp)}, coverage={compute_coverage(adj, S_ilp):.3f}")
    print(f"Greedy: |D|={len(S_gr)}, coverage={compute_coverage(adj, S_gr):.3f}")
    method_suffix = get_completion_method_suffix(GNN_COMPLETION_MODE)
    print(f"{model_name}+{method_suffix}: |D|={len(S_gg)}, coverage={compute_coverage(adj, S_gg):.3f}")

    result_sets = {
        "ILP": S_ilp,
        "Greedy": S_gr,
        f"{model_name}+PruneGuidedGreedy": S_gg,
    }
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(EXPERIMENT_RESULTS_DIR, "Figures_size_all_holes"), exist_ok=True)
    save_path = os.path.join(EXPERIMENT_RESULTS_DIR, "Figures_size_all_holes", f"compare_8models_{timestamp}.png")

    plot_methods_nodes_edges(
        adj,
        coords,
        result_sets,
        main_title=f"One GNN vs ILP&Greedy  ({m}×{n}, hole={hole_ratio:.2f})",
        save_path=save_path,
        m=m,
        n=n,
        hole_rate=hole_ratio,
        show_grid=True,
        show_all_methods=True,
    )


# =========================================================
#  功能 4：一次測試 7 個模型
# =========================================================

def test_all_7_GNNmodels():
    global GLOBAL_MODELS, GLOBAL_DEVICE
    if not GLOBAL_MODELS or GLOBAL_DEVICE is None:
        print("❌ 尚未訓練模型")
        return

    print("\n=== 測試 6 種 GNN 模型（Irregular 測試圖 + GuidedGreedy）===")

    m = input_int_with_default("m", 20, min_value=1)
    n = input_int_with_default("n", 20, min_value=1)
    hole_ratio = input_float_with_default("不規則挖洞比例 hole_ratio (例如 0.2，直接 Enter = 0)", 0.0, min_value=0.0, max_value=0.95)

    global GLOBAL_M, GLOBAL_N, GLOBAL_HOLE
    GLOBAL_M = m
    GLOBAL_N = n
    GLOBAL_HOLE = hole_ratio

    adj, coords, hole_ratio = build_irregular_grid_adj(m, n, hole_ratio=hole_ratio, ensure_connected=True)
    data = build_full_features(m, n, adj, coords=coords, pe_dim=8, rwe_dim=16).to(GLOBAL_DEVICE)

    S_ilp = ilp_minimum_dominating_set(adj)
    S_gr = greedy_dominating_set(adj)

    result_sets = {
        "ILP": S_ilp,
        "Greedy": S_gr
    }

    model_types = ["GCN", "GATv2", "SAGE", "GIN", "TRANSFORMER", "GRAPHORMER", "GPSCONV"]

    for mt in model_types:
        if mt in GLOBAL_MODELS:
            print(f"[Evaluating {mt} + PruneGuidedGreedy]")
            model = GLOBAL_MODELS[mt]
            model.eval()
            model_in_dim = GLOBAL_MODEL_IN_DIM or _infer_model_in_dim(model)
            x_eval = _pad_or_trunc_x(data.x, model_in_dim).to(GLOBAL_DEVICE)
            N = x_eval.size(0)
            batch = torch.zeros(N, dtype=torch.long, device=GLOBAL_DEVICE)
            with torch.no_grad():
                logits, g_emb = model(x_eval, data.edge_index, batch=batch)
                probs = torch.sigmoid(logits)
            S_gg = gnn_raw_then_complete(adj, probs, threshold=0.5, ilp_cutoff=ILP_CompleteTime, completion_mode=GNN_COMPLETION_MODE)
            result_sets[f"{mt}+PruneGuidedGreedy"] = S_gg
        else:
            print(f"⚠ 模型 {mt} 尚未訓練 → 跳過")

    for name, S in result_sets.items():
        cov = compute_coverage(adj, S)
        print(f"{name}: |D|={len(S)})")#, coverage={cov:.3f}"

        import os, datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(EXPERIMENT_RESULTS_DIR, "Figures_size_all_holes"), exist_ok=True)
    save_path = os.path.join(EXPERIMENT_RESULTS_DIR, "Figures_size_all_holes", f"compare_8models_{timestamp}.png")

    plot_methods_nodes_edges(
        adj,
        coords,
        result_sets,
        main_title=f"Six Methods Comparison  ({m}×{n}, hole={hole_ratio:.2f})",
        save_path=save_path,
        m=m,
        n=n,
        hole_rate=hole_ratio,
        show_grid=True,
        show_all_methods=True,
    )


# =========================================================
#  功能 12：一次測試 ILP+Greedy + 1種 選取的GNN模型 並比較
# =========================================================


def test_all_3_methods():
    global GLOBAL_MODELS, GLOBAL_DEVICE
    if not GLOBAL_MODELS or GLOBAL_DEVICE is None:
        print("❌ 尚未訓練/載入模型。")
        return

    print("\n=== 一次測試 3 個方法：ILP + Greedy + 1 個 GNN模型 ===")
    print("可用模型：", list(GLOBAL_MODELS.keys()))
    model_name = list(GLOBAL_MODELS.keys())[0]
    print(f"[Test-3Methods] 使用模型：{model_name}")

    m = input_int_with_default("m", 20, min_value=1)
    n = input_int_with_default("n", 20, min_value=1)
    hole_ratio = input_float_with_default("不規則挖洞比例 hole_ratio (例如 0.2，直接 Enter = 0)", 0.0, min_value=0.0, max_value=0.95)

    global GLOBAL_M, GLOBAL_N, GLOBAL_HOLE
    GLOBAL_M = m
    GLOBAL_N = n
    GLOBAL_HOLE = hole_ratio

    adj, coords, hole_ratio = build_irregular_grid_adj(m, n, hole_ratio=hole_ratio, ensure_connected=True)
    if not adj:
        print("❌ 生成圖失敗或圖為空。")
        return

    data = build_full_features(m, n, adj, coords=coords, pe_dim=8, rwe_dim=16).to(GLOBAL_DEVICE)

    model = GLOBAL_MODELS[model_name]
    model.eval()
    model_in_dim = GLOBAL_MODEL_IN_DIM or _infer_model_in_dim(model)
    data.x = _pad_or_trunc_x(data.x, model_in_dim).to(GLOBAL_DEVICE)

    N = data.x.size(0)
    batch = torch.zeros(N, dtype=torch.long, device=GLOBAL_DEVICE)
    with torch.no_grad():
        logits, _ = model(data.x, data.edge_index, batch=batch)
        probs = torch.sigmoid(logits)

    S_ilp, ilp_info = ilp_minimum_dominating_set_with_info(adj, time_limit=ILP_MaxTime)
    S_gr = greedy_dominating_set(adj)
    S_gnn = gnn_raw_then_complete(
        adj, probs, threshold=0.5,
        ilp_cutoff=ILP_CompleteTime,
        completion_mode=GNN_COMPLETION_MODE
    )

    cov_ilp = compute_coverage(adj, S_ilp)
    cov_gr = compute_coverage(adj, S_gr)
    cov_gnn = compute_coverage(adj, S_gnn)

    print("\n=== 比較 ===")
    print(f"ILP: |D|={len(S_ilp)}, coverage={cov_ilp:.3f}")
    print(f"     {_format_ilp_info_short(ilp_info)}")
    print(f"Greedy: |D|={len(S_gr)}, coverage={cov_gr:.3f}")
    method_suffix = get_completion_method_suffix(GNN_COMPLETION_MODE)
    print(f"{model_name}+{method_suffix}: |D|={len(S_gnn)}, coverage={cov_gnn:.3f}")

    gap_greedy_vs_ilp = len(S_gr) - len(S_ilp)
    gap_gnn_vs_ilp = len(S_gnn) - len(S_ilp)
    ilp_ref_name = "ILP-opt" if bool(ilp_info.get("optimal", False)) else "ILP-feasible"
    print(f"[Gap] Greedy - {ilp_ref_name} = {gap_greedy_vs_ilp:+d}")
    print(f"[Gap] {model_name}+{method_suffix} - {ilp_ref_name} = {gap_gnn_vs_ilp:+d}")

    result_sets = {
        f"ILP ({'OPT' if bool(ilp_info.get('optimal', False)) else 'FEAS'})": S_ilp,
        "Greedy": S_gr,
        f"{model_name}+{method_suffix}": S_gnn,
    }

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(EXPERIMENT_RESULTS_DIR, "Figures_size_all_holes")
    os.makedirs(out_dir, exist_ok=True)

    ilp_title_tag = "OPT" if bool(ilp_info.get("optimal", False)) else str(ilp_info.get("status_class", "UNK"))
    main_title = (
        f"3-Method Comparison ({model_name}) | "
        f"{m}x{n}, hole={float(hole_ratio):.2f} | "
        f"ILP={ilp_title_tag}, t={float(ilp_info.get('time_sec', 0.0) or 0.0):.3f}s"
    )

    save_path = os.path.join(out_dir, f"compare_3methods_{model_name}_{timestamp}.png")
    plot_methods_nodes_edges(
        adj=adj,
        coords=coords,
        result_sets=result_sets,
        main_title=main_title,
        save_path=save_path,
        m=m,
        n=n,
        hole_rate=hole_ratio,
        show_grid=True,
        show_all_methods=True,
    )
    print(f"[Plot] saved -> {save_path}")

    csv_dir = os.path.join(EXPERIMENT_RESULTS_DIR, "ThreeMethodEval")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"compare_3methods_{model_name}_{timestamp}.csv")
    row = {
        "model": model_name,
        "m": int(m),
        "n": int(n),
        "hole_ratio": float(hole_ratio),
        "N_nodes": int(len(adj)),
        "ilp_size": int(len(S_ilp)),
        "ilp_coverage": float(cov_ilp),
        "ilp_status": str(ilp_info.get("status", "")),
        "ilp_status_class": str(ilp_info.get("status_class", "")),
        "ilp_optimal": int(bool(ilp_info.get("optimal", False))),
        "ilp_success": int(bool(ilp_info.get("success", False))),
        "ilp_attempted": int(bool(ilp_info.get("attempted", False))),
        "ilp_fallback": int(bool(ilp_info.get("fallback", False))),
        "ilp_time_sec": float(ilp_info.get("time_sec", 0.0) or 0.0),
        "greedy_size": int(len(S_gr)),
        "greedy_coverage": float(cov_gr),
        "gnn_size": int(len(S_gnn)),
        "gnn_coverage": float(cov_gnn),
        "completion_mode": str(GNN_COMPLETION_MODE),
        "gap_greedy_vs_ilp_ref": int(gap_greedy_vs_ilp),
        "gap_gnn_vs_ilp_ref": int(gap_gnn_vs_ilp),
        "ilp_gap_ref": "ILP_OPT" if bool(ilp_info.get("optimal", False)) else "ILP_FEASIBLE",
        "figure_path": save_path,
    }
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"[CSV] saved -> {csv_path}")


def run_experiment_multi_size_and_holes():
    global GLOBAL_MODELS, GLOBAL_DEVICE

    if not GLOBAL_MODELS or GLOBAL_DEVICE is None:
        print("❌ 尚未訓練或載入模型，請先執行選單 4 或 9。")
        return

    os.makedirs(EXPERIMENT_RESULTS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(
        EXPERIMENT_RESULTS_DIR,
        f"experiment_multiSize_multiHole_{timestamp}.csv"
    )

    header = [
        "m", "n", "N_nodes", "hole_ratio",
        "topology",
        "method",
        "set_size",
        "coverage",
        "time_sec",
    ]

    print("\n=== 實驗模式：多尺寸 + 多挖洞比例，自動比較並輸出 CSV ===")
    print(f"Grid sizes = {EXPERIMENT_GRID_SIZES}")
    print(f"Hole rates = {EXPERIMENT_HOLE_RATES}")
    print(f"結果將輸出至：{csv_path}")

    # 新增：詢問每個 (m,n,hole_ratio) 要重複幾張圖
    repeat_k = input_int_with_default(
        "每個 (m,n,hole_ratio) 要重複幾張不同隨機圖？", 3
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for (m, n) in EXPERIMENT_GRID_SIZES:
            for hole_ratio_target in EXPERIMENT_HOLE_RATES:
                for rep in range(1, repeat_k + 1):
                    print(
                        f"\n[Experiment] m={m}, n={n}, "
                        f"hole_ratio={hole_ratio_target:.2f}, "
                        f"repeat={rep}/{repeat_k}"
                    )

                    # --- 建圖（隨機挖洞） ---
                    t0 = time.perf_counter()
                    adj, coords, hole_ratio = build_irregular_grid_adj(
                        m, n,
                        hole_ratio=hole_ratio_target,  # 使用目標挖洞比例，實際比例回傳在 hole_ratio
                        ensure_connected=True
                    )
                    build_time = time.perf_counter() - t0

                    N = len(adj)
                    data = build_full_features(
                        m, n,
                        adj,
                        coords=coords,
                        pe_dim=8,
                        rwe_dim=16
                    ).to(GLOBAL_DEVICE)

                    # --- ILP baseline ---
                    t1 = time.perf_counter()
                    try:
                        S_ilp = ilp_minimum_dominating_set(adj)
                    except Exception as e:
                        print(f"[ILP] 失敗：{e}")
                        S_ilp = []
                    ilp_time = time.perf_counter() - t1 + build_time
                    cov_ilp = compute_coverage(adj, S_ilp)
                    writer.writerow([
                        m, n, N, hole_ratio,
                        GRAPH_TOPOLOGY,
                        "ILP",
                        len(S_ilp),
                        cov_ilp,
                        ilp_time,
                    ])

                    # --- Greedy baseline ---
                    t2 = time.perf_counter()
                    S_gr = greedy_dominating_set(adj)
                    greedy_time = time.perf_counter() - t2 + build_time
                    cov_gr = compute_coverage(adj, S_gr)
                    writer.writerow([
                        m, n, N, hole_ratio,
                        GRAPH_TOPOLOGY,
                        "Greedy",
                        len(S_gr),
                        cov_gr,
                        greedy_time,
                    ])

                    # --- 7 個 GNN + GuidedGreedy ---
                    model_types = [
                        "GCN", "GATv2", "SAGE",
                        "GIN", "TRANSFORMER",
                        "GRAPHORMER", "GPSCONV"
                    ]

                    for mt in model_types:
                        if mt not in GLOBAL_MODELS:
                            print(f"⚠ 模型 {mt} 尚未訓練 → 跳過")
                            continue

                        model = GLOBAL_MODELS[mt]
                        model.eval()
                        batch = torch.zeros(
                            data.x.size(0),
                            dtype=torch.long,
                            device=GLOBAL_DEVICE
                        )

                        t3 = time.perf_counter()
                        with torch.no_grad():
                            logits, g_emb = model(
                                data.x,
                                data.edge_index,
                                batch=batch
                            )
                            probs = torch.sigmoid(logits)
                            S_gg = gnn_raw_then_complete(
                                adj, probs,
                                threshold=PROB_SOFTREPAIR_THRESHOLD,
                                ilp_cutoff=ILP_CompleteTime,
                                beta=PROB_SOFTREPAIR_BETA,
                                verbose=PROB_SOFTREPAIR_VERBOSE,
                                completion_mode=GNN_COMPLETION_MODE
                            )

                        eval_time = time.perf_counter() - t3 + build_time
                        cov = compute_coverage(adj, S_gg)

                        method_name = f"{mt}+{get_completion_method_suffix(GNN_COMPLETION_MODE)}"
                        writer.writerow([
                            m, n, N, hole_ratio,
                            GRAPH_TOPOLOGY,
                            method_name,
                            len(S_gg),
                            cov,
                            eval_time,
                        ])

                        print(
                            f"  - {method_name}: "
                            f"|D|={len(S_gg)}, time={eval_time:.3f}s"
                            # f", coverage={cov:.3f}"
                        )

    print("\n✔ 實驗完成，CSV 已輸出到：", csv_path)


"""
def run_experiment_multi_size_and_holes():
    global GLOBAL_MODELS, GLOBAL_DEVICE

    if not GLOBAL_MODELS or GLOBAL_DEVICE is None:
        print("❌ 尚未訓練或載入模型，請先執行選單 4 或 9。")
        return

    os.makedirs(EXPERIMENT_RESULTS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(
        EXPERIMENT_RESULTS_DIR,
        f"experiment_multiSize_multiHole_{timestamp}.csv"
    )

    header = [
        "m", "n", "N_nodes", "hole_ratio",
        "topology",
        "method",
        "set_size",
        "coverage",
        "time_sec",
    ]

    print("\n=== 實驗模式：多尺寸 + 多挖洞比例，自動比較並輸出 CSV ===")
    print(f"Grid sizes = {EXPERIMENT_GRID_SIZES}")
    print(f"Hole rates = {EXPERIMENT_HOLE_RATES}")
    print(f"結果將輸出至：{csv_path}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for (m, n) in EXPERIMENT_GRID_SIZES:
            for hole_ratio in EXPERIMENT_HOLE_RATES:
                print(f"\n[Experiment] m={m}, n={n}, hole_ratio={hole_ratio:.2f}")

                t0 = time.perf_counter()
                adj, coords, hole_ratio = build_irregular_grid_adj(
                    m, n,
                    hole_ratio=hole_ratio,
                    ensure_connected=True
                )
                build_time = time.perf_counter() - t0

                N = len(adj)
                data = build_full_features(
                    m, n,
                    adj,
                    coords=coords,
                    pe_dim=8,
                    rwe_dim=16
                ).to(GLOBAL_DEVICE)

                t1 = time.perf_counter()
                try:
                    S_ilp = ilp_minimum_dominating_set(adj)
                except Exception as e:
                    print(f"[ILP] 失敗：{e}")
                    S_ilp = []
                ilp_time = time.perf_counter() - t1 + build_time
                cov_ilp = compute_coverage(adj, S_ilp)
                writer.writerow([
                    m, n, N, hole_ratio,
                    GRAPH_TOPOLOGY,
                    "ILP",
                    len(S_ilp),
                    cov_ilp,
                    ilp_time,
                ])

                t2 = time.perf_counter()
                S_gr = greedy_dominating_set(adj)
                greedy_time = time.perf_counter() - t2 + build_time
                cov_gr = compute_coverage(adj, S_gr)
                writer.writerow([
                    m, n, N, hole_ratio,
                    GRAPH_TOPOLOGY,
                    "Greedy",
                    len(S_gr),
                    cov_gr,
                    greedy_time,
                ])

                model_types = ["GCN", "GATv2", "SAGE", "GIN", "TRANSFORMER", "GRAPHORMER", "GPSCONV"]

                for mt in model_types:
                    if mt not in GLOBAL_MODELS:
                        print(f"⚠ 模型 {mt} 尚未訓練 → 跳過")
                        continue

                    model = GLOBAL_MODELS[mt]
                    model.eval()
                    batch = torch.zeros(
                        data.x.size(0),
                        dtype=torch.long,
                        device=GLOBAL_DEVICE
                    )

                    t3 = time.perf_counter()
                    with torch.no_grad():
                        logits, g_emb = model(
                            data.x,
                            data.edge_index,
                            batch=batch
                        )
                        probs = torch.sigmoid(logits)
                        S_gg = gnn_raw_then_complete(adj, probs, threshold=0.5, ilp_cutoff=100, completion_mode=GNN_COMPLETION_MODE)

                    eval_time = time.perf_counter() - t3 + build_time
                    cov = compute_coverage(adj, S_gg)

                    method_name = f"{mt}+{get_completion_method_suffix(GNN_COMPLETION_MODE)}"
                    writer.writerow([
                        m, n, N, hole_ratio,
                        GRAPH_TOPOLOGY,
                        method_name,
                        len(S_gg),
                        cov,
                        eval_time,
                    ])

                    print(
                        f"  - {method_name}: "
                        f"|D|={len(S_gg)}, time={eval_time:.3f}s"#, coverage={cov:.3f}
                    )

    print("\n✔ 實驗完成，CSV 已輸出到：", csv_path)
"""

def menu_set_topology():
    global GRAPH_TOPOLOGY
    print("\n=== 設定圖形拓樸 (Topology) ===")
    print("1. grid (4 方向)")
    print("2. supergrid (8 方向，含對角線)")
    choice = input("請選擇：").strip()

    if choice == "1":
        GRAPH_TOPOLOGY = "grid"
    elif choice == "2":
        GRAPH_TOPOLOGY = "supergrid"
    else:
        print("❌ 無效選擇，維持原設定。")

    print(f"✔ 已設定拓樸為：{GRAPH_TOPOLOGY}")
# =========================================================

# =========================================================
#  實驗結果分析與繪圖：讀取 CSV，自動產生圖與排名表
# =========================================================

def _smart_label_offset(idx):
    offsets=[(0,6),(0,-6),(6,0),(-6,0),(6,6),(-6,-6)]
    return offsets[idx%len(offsets)]

# =========================================================
# 同一尺寸 × 所有挖洞比例：左右兩子圖（|D|、time_sec）
# =========================================================
def plot_all_methods_one_size_all_holes_two_subplots(combos, sizes, holes, methods_set):
    import matplotlib.pyplot as plt

    from pathlib import Path, os
    from matplotlib.transforms import offset_copy

    out_dir = os.path.join(EXPERIMENT_RESULTS_DIR, "Figures_size_all_holes")
    os.makedirs(out_dir, exist_ok=True)

    base_colors = [
        "black","red","blue","green","orange",
        "purple","brown","cyan","magenta","olive","navy","teal"
    ]
    methods = sorted(list(methods_set))
    color_map = {m: base_colors[i % len(base_colors)] for i, m in enumerate(methods)}

    for (m, n) in sizes:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 左子圖：|D|
        for method in methods:
            xs, ys = [], []
            for h in holes:
                key = (m, n, h)
                recs = combos.get(key, [])
                r0 = next((r for r in recs if r["method"] == method), None)
                if r0:
                    xs.append(h)
                    ys.append(r0["set_size"])

            if xs:
                ax1.plot(xs, ys, "o-", color=color_map[method], label=method)
                for idx, (x, y) in enumerate(zip(xs, ys)):
                    dx, dy = _smart_label_offset(idx)
                    text = ax1.text(x, y, f"{int(y)}", fontsize=8, ha="center", va="bottom")
                    text.set_transform(offset_copy(ax1.transData, fig=fig, x=dx, y=dy, units="points"))

        ax1.set_title(f"|D| (Dominating set size) (max size={m}x{n})")
        ax1.set_xlabel("hole ratio")
        ax1.set_ylabel("|D| (size of dominating set)")
        ax1.grid(True, ls="--", alpha=0.4)

        # 右子圖：time_sec
        for method in methods:
            xs, ys = [], []
            for h in holes:
                key = (m, n, h)
                recs = combos.get(key, [])
                r0 = next((r for r in recs if r["method"] == method), None)
                if r0:
                    xs.append(h)
                    ys.append(r0["time_sec"])

            if xs:
                ax2.plot(xs, ys, "o-", color=color_map[method], label=method)
                for idx, (x, y) in enumerate(zip(xs, ys)):
                    dx, dy = _smart_label_offset(idx)
                    text = ax2.text(x, y, f"{y:.2f}", fontsize=8, ha="center", va="bottom")
                    text.set_transform(offset_copy(ax2.transData, fig=fig, x=dx, y=dy, units="points"))

        ax2.set_title(f"Computation time (max size={m}x{n})")
        ax2.set_xlabel("hole ratio")
        ax2.set_ylabel("computing time (sec)")
        ax2.grid(True, ls="--", alpha=0.4)

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02))

        fig.tight_layout()
        fname = os.path.join(out_dir, f"size_{m}x{n}_all_holes.png")
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"✔ 已完成尺寸 {m}x{n} 的左右雙子圖：{fname}")

# =========================================================
def plot_two_big_figures_S_time_C_smart(combos, sizes, holes, methods_set):
    import matplotlib.pyplot as plt

    from pathlib import Path, os
    from matplotlib.transforms import offset_copy

    out_dir = os.path.join(EXPERIMENT_RESULTS_DIR, "Figures_size_all_holes")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # 固定每個 method 的顏色
    base_colors = [
        "black", "red", "blue", "green", "orange",
        "purple", "brown", "cyan", "magenta", "olive", "navy", "teal"
    ]
    methods = sorted(list(methods_set))
    color_map = {m: base_colors[i % len(base_colors)] for i, m in enumerate(methods)}

    # =========================================================
    # 1) |D| vs hole_ratio
    # =========================================================
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for method in methods:
        for (m, n) in sizes:
            xs, ys = [], []
            for h in holes:
                key = (m, n, h)
                rlist = combos.get(key, [])
                r0 = next((r for r in rlist if r["method"] == method), None)
                if r0:
                    xs.append(h)
                    ys.append(r0["set_size"])
            if xs:
                ax.plot(xs, ys, "o-", color=color_map[method],
                        label=f"{method}（{m}×{n}）")

                # 數值標籤 + smart offset（使用 transforms.offset_copy）
                for idx, (x, y) in enumerate(zip(xs, ys)):
                    dx, dy = _smart_label_offset(idx)
                    text = ax.text(x, y, f"{int(y)}",
                            fontsize=7, ha="center", va="bottom")
                    text.set_transform(
                        offset_copy(ax.transData, fig=plt.gcf(),
                                    x=dx, y=dy, units="points")
                    )

    ax.set_xlabel("hole ratio (挖洞比例)")
    ax.set_ylabel("|D| (支配集大小)")
    ax.set_title(
    "|D| vs hole_ratio (all methods x all experiment sizes)\n"
    f"(Domination, {GLOBAL_M}x{GLOBAL_N}, hole_rate={(GLOBAL_HOLE if GLOBAL_HOLE is not None else 0):.2f}, topology={GLOBAL_TOPOLOGY})",
    fontsize=9)
    ax.grid(True, ls="--", alpha=0.4)
    #ax.legend(fontsize=6, ncol=2)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        fontsize=6    )
    plt.tight_layout()
    safe_method = str(locals().get("main_title", "summary")).replace(" ", "_").replace("(", "").replace(")", "")
    filename = f"compare_m{GLOBAL_M}_n{GLOBAL_N}_h{(GLOBAL_HOLE if GLOBAL_HOLE is not None else 0):.2f}_{GLOBAL_TOPOLOGY}_{safe_method}.png"
    plt.savefig(os.path.join(out_dir, filename), dpi=300)
    print(f"[Saved] {filename}")
    plt.close()

    # =========================================================
    # 2) time_sec vs hole_ratio
    # =========================================================
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for method in methods:
        for (m, n) in sizes:
            xs, ys = [], []
            for h in holes:
                key = (m, n, h)
                rlist = combos.get(key, [])
                r0 = next((r for r in rlist if r["method"] == method), None)
                if r0:
                    xs.append(h)
                    ys.append(r0["time_sec"])
            if xs:
                ax.plot(xs, ys, "o-", color=color_map[method],
                        label=f"{method}（{m}×{n}）")

                for idx, (x, y) in enumerate(zip(xs, ys)):
                    dx, dy = _smart_label_offset(idx)
                    text = ax.text(x, y, f"{y:.2f}",
                            fontsize=7, ha="center", va="bottom")
                    text.set_transform(
                        offset_copy(ax.transData, fig=plt.gcf(),
                                    x=dx, y=dy, units="points")
                    )

    ax.set_xlabel("hole_ratio")
    ax.set_ylabel("time_sec")
    ax.set_title(
    "|D| vs hole_ratio (all methods x all experiment sizes)\n"
    f"(Domination, {GLOBAL_M}×{GLOBAL_N}, hole_rate={(GLOBAL_HOLE if GLOBAL_HOLE is not None else 0):.2f}, topology={GLOBAL_TOPOLOGY})",
    fontsize=9)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    safe_method = str(locals().get("main_title", "summary")).replace(" ", "_").replace("(", "").replace(")", "")
    filename = f"compare_m{GLOBAL_M}_n{GLOBAL_N}_h{(GLOBAL_HOLE if GLOBAL_HOLE is not None else 0):.2f}_{GLOBAL_TOPOLOGY}_{safe_method}.png"
    plt.savefig(os.path.join(out_dir, filename), dpi=300)
    print(f"[Saved] {filename}")
    plt.close()

def analyze_and_plot_experiment_results():
    """
    讀取 experiment_results/ 內最新的實驗 CSV，自動：
    1) 建立排名表 (依 |D| 由小到大, |D|為支配集 : 此功能目前不製作)
    2) 輸出 summary CSV
    3) 產生數張比較圖 (heatmap + 折線圖) : heapmap 不做, 每個方法固定mxn下, 依hole rate vs |S| 來製作折線圖
    """
    if not os.path.isdir(EXPERIMENT_RESULTS_DIR):
        print(f"❌ 找不到實驗資料夾：{EXPERIMENT_RESULTS_DIR}")
        return

    csv_files = [
        f for f in os.listdir(EXPERIMENT_RESULTS_DIR)
        if f.endswith(".csv")
    ]
    if not csv_files:
        print(f"❌ 在 {EXPERIMENT_RESULTS_DIR} 中找不到任何 CSV 結果檔。請先執行選單 10 產生實驗結果。")
        return

    csv_files.sort()
    latest = csv_files[-1]
    csv_path = os.path.join(EXPERIMENT_RESULTS_DIR, latest)
    print(f"✔ 將使用最新實驗檔案：{csv_path}")

    # 讀取 CSV
    combos = {}   # key = (m,n,hole_ratio) -> list of dict
    methods_set = set()
    sizes_set = set()
    hole_set = set()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                m = int(row["m"])
                n = int(row["n"])
                N_nodes = int(row.get("N_nodes", m * n))
                hole_ratio = float(row["hole_ratio"])
                method = row["method"]
                set_size = float(row["set_size"])
                coverage = float(row["coverage"])
                time_sec = float(row.get("time_sec", 0.0))
            except Exception as e:
                print(f"[警告] 讀取某列失敗：{e} → 略過該列")
                continue

            key = (m, n, hole_ratio)
            rec = {
                "m": m,
                "n": n,
                "N_nodes": N_nodes,
                "hole_ratio": hole_ratio,
                "method": method,
                "set_size": set_size,
                "coverage": coverage,
                "time_sec": time_sec,
            }
            combos.setdefault(key, []).append(rec)
            methods_set.add(method)
            sizes_set.add((m, n))
            hole_set.add(hole_ratio)

    if not combos:
        print("❌ 沒有任何有效資料可供分析。")
        return

    sizes = sorted(list(sizes_set))
    holes = sorted(list(hole_set))

    print(f"✔ 共讀入 {len(combos)} 組 (size 支配數, hole_ratio 挖洞例) 實驗結果。")
    print(f"  - sizes = {sizes}")
    print(f"  - hole_ratios = {holes}")
    print(f"  - methods = {sorted(list(methods_set))}")

    # -----------------------------------------------------
    # 1) 產生排名表並輸出 summary CSV
    # -----------------------------------------------------
    summary_rows = []
    for (m, n, hole_ratio), recs in combos.items():
        # 依 |S| 由小到大， time 由小到大 排序
        recs_sorted = sorted(
            recs,
            key=lambda r: (r["set_size"], -r["coverage"], r["time_sec"])
        )
        for rank, r in enumerate(recs_sorted[:3], start=1):
            summary_rows.append({
                "m": m,
                "n": n,
                "N_nodes": r["N_nodes"],
                "hole_ratio": hole_ratio,
                "rank": rank,
                "method": r["method"],
                "set_size": r["set_size"],
                "coverage": r["coverage"], #coverage 去掉此參數
                "time_sec": r["time_sec"],
            })

    # 輸出 summary CSV
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(
        EXPERIMENT_RESULTS_DIR,
        f"summary_multiSize_multiHole_{ts}.csv"
    )
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "m", "n", "N_nodes", "hole_ratio",
            "rank", "method", "set_size", "coverage", "time_sec"#coverage 去掉此參數
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"✔ 已輸出 summary CSV：{summary_path}")
    print("  排名規則：先比 |D| 較小，其次 coverage 較大，其次 time 較短。")#coverage 去掉此參數

    # -----------------------------------------------------
    # heatmap & coverage graphs removed
    # -----------------------------------------------------
    plot_two_big_figures_S_time_C_smart(combos, sizes, holes, methods_set)
    plot_all_methods_one_size_all_holes_two_subplots(combos, sizes, holes, methods_set)
    print("✔ 實驗結果分析與繪圖完成.")


# =========================================================
#  主選單
# =========================================================

############################################################
# Improved v3.5: Smart Offset for Label Overlap Avoidance
############################################################
def smart_offset(idx):
    offsets = [(0,6),(0,-6),(6,0),(-6,0),(6,6),(-6,6)]
    return offsets[idx % len(offsets)]

def plot_two_big_figures_S_time_C_smart(
    combos, sizes, holes, methods_set,
    main_title="summary",
    out_subdir="Figures_size_all_holes",
    dpi=300
):
    import os
    import matplotlib.pyplot as plt

    from pathlib import Path

    # ---- folders ----
    dest_dir = os.path.join(EXPERIMENT_RESULTS_DIR, out_subdir)
    os.makedirs(dest_dir, exist_ok=True)

    # ---- fixed colors ----
    color_map = [
        "black","red","blue","green","orange","purple","brown",
        "cyan","magenta","olive","darkred","navy","teal"
    ]
    method_list = sorted(list(methods_set))
    method_color = {m: color_map[i % len(color_map)] for i, m in enumerate(method_list)}

    holes_sorted = sorted(holes)

    def _safe(s: str) -> str:
        return str(s).replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")

    safe_title = _safe(main_title)

    def _plot_metric(metric_key, y_label, fig_title, y_fmt, file_suffix):
        """
        metric_key: "set_size" or "time_sec"
        y_fmt: callable y->str
        file_suffix: distinguish filenames to avoid overwrite
        """
        plt.figure(figsize=(10, 6))

        # 全域 offset index（跨 method / size / hole）
        idx_global = 0

        for method in method_list:
            for (m, n) in sizes:
                xs, ys = [], []
                for hole in holes_sorted:
                    key = (m, n, hole)
                    if key not in combos:
                        continue
                    recs = combos[key]
                    found = [r for r in recs if r.get("method") == method]
                    if found:
                        xs.append(hole)
                        ys.append(found[0].get(metric_key))

                if xs:
                    label = f"{method} ({m}x{n})"
                    plt.plot(xs, ys, marker="o", color=method_color[method], label=label)

                    for x, y in zip(xs, ys):
                        # y 可能是 None
                        if y is None:
                            continue
                        dx, dy = smart_offset(idx_global)
                        idx_global += 1
                        plt.annotate(
                            y_fmt(y),
                            xy=(x, y),
                            fontsize=7,
                            ha="center",
                            va="bottom",
                            xytext=(dx, dy),
                            textcoords="offset points"
                        )

        plt.xlabel("hole ratio")
        plt.ylabel(y_label)
        plt.title(fig_title)
        plt.grid(True, linestyle="--", alpha=0.4)

        # legend 欄數動態，避免太擠
        ncol = 3 if len(method_list) * max(1, len(sizes)) <= 18 else 4
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=ncol,
            fontsize=6
        )
        plt.tight_layout()

        filename = (
            f"compare_{file_suffix}_"
            f"m{GLOBAL_M}_n{GLOBAL_N}_h{(GLOBAL_HOLE if GLOBAL_HOLE is not None else 0):.2f}_"
            f"{GLOBAL_TOPOLOGY}_{safe_title}.png"
        )
        plt.savefig(os.path.join(dest_dir, filename), dpi=dpi)
        print(f"[Saved] {os.path.join(dest_dir, filename)}")
        plt.close()

    # --- FIGURE 1: |D| ---
    _plot_metric(
        metric_key="set_size",
        y_label="|D| (size of dominating set)",
        fig_title="|D| (Dominating set cardinality)",
        y_fmt=lambda y: f"{int(y)}" if isinstance(y, (int, float)) else str(y),
        file_suffix="SIZE"
    )

    # --- FIGURE 2: time_sec ---
    _plot_metric(
        metric_key="time_sec",
        y_label="time (sec)",
        fig_title="Computation Time",
        y_fmt=lambda y: f"{float(y):.2f}",
        file_suffix="TIME"
    )


    # Smart offset version
    # ============================
# Smart total curve plots v3.7
# ============================

# =========================================================
#  TrainSafe Manager – 核心工具 (簡化版)
# =========================================================

def pad_features_to_dim(x, target_dim):
    """將 feature 向量 padding 或截斷為 target_dim。"""
    N, old_dim = x.size()
    if old_dim == target_dim:
        return x

    if old_dim < target_dim:
        pad = torch.zeros((N, target_dim - old_dim), dtype=x.dtype)
        return torch.cat([x, pad], dim=1)

    # old_dim > target_dim → 截斷
    return x[:, :target_dim]


def analyze_train_safe_Connected_folder(folder):
    graphs = load_training_graphs_safe(folder)
    if not graphs:
        return {"error": "no graphs"}

    dims = set()
    bad = 0

    for i, g in enumerate(graphs):
        try:
            data = g.get("data", None) if isinstance(g, dict) else None
            if data is None:
                bad += 1
                print(f"[TrainSafe][Analyze] graph#{i} has no 'data'. type={type(g)} keys={list(g.keys()) if isinstance(g, dict) else 'NA'}")
                continue

            # PyG Data: prefer attribute, fallback to dict-style
            if hasattr(data, "x") and data.x is not None:
                x = data.x
            elif isinstance(data, dict) and "x" in data:
                x = data["x"]
            else:
                bad += 1
                print(f"[TrainSafe][Analyze] graph#{i} data has no x. data_type={type(data)} data_keys={list(data.keys()) if isinstance(data, dict) else getattr(data, 'keys', lambda: 'NA')()}")
                continue

            dims.add(int(x.shape[1]))
        except Exception as e:
            bad += 1
            print(f"[TrainSafe][Analyze] graph#{i} failed: {e}")

    rep = {
        "unique_dims": sorted(list(dims)),
        "health": (len(dims) == 1 and bad == 0),
        "bad_graphs": bad, 
        "edge_health": True,   # ← 加這行
    }
    return rep


def batch_auto_fix_train_safe(feature_fix=True, edge_fix_mode="shape_then_rebuild"):
    """
    簡化版：只印出報告，不進行實際修補，但保留介面。
    """
    folders = scan_train_safe_Connected_folders()
    print("\n[TrainSafe] Auto-fix (報告模式)")
    if not folders:
        print("找不到 Train_<topology>_* 資料夾。")
        return

    for folder in folders:
        rep = analyze_train_safe_Connected_folder(folder)
        if "error" in rep:
            print(f"  - {folder}: ❌ {rep['error']}")
        else:
            print(f"  - {folder}: 檔案數={rep['count']}, feature_dims={rep['unique_dims']}, feature健康={rep['health']}, edge健康={rep['edge_health']}")


class TrainSafeManagerGUI:
    """
    簡化版 GUI:
        - 列出所有 Train_<topology>_* 資料夾
        - 顯示每個資料夾的基本健康度
    """

    def __init__(self, master):
        self.master = master
        master.title("TrainSafe Manager (簡化版)")

        self.frame = ttk.Frame(master, padding=10)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(self.frame, columns=("count", "dims", "health", "edge"), show="headings")
        self.tree.heading("count", text="檔案數")
        self.tree.heading("dims", text="feature_dims")
        self.tree.heading("health", text="feature健康")
        self.tree.heading("edge", text="edge健康")
        self.tree.pack(fill=tk.BOTH, expand=True)

        btn_frame = ttk.Frame(self.frame)
        btn_frame.pack(fill=tk.X, pady=5)

        btn_refresh = ttk.Button(btn_frame, text="重新整理", command=self.refresh)
        btn_refresh.pack(side=tk.LEFT, padx=5)

        info_label = ttk.Label(btn_frame, text="※ 本版為報告模式，不修改檔案")
        info_label.pack(side=tk.LEFT, padx=5)

        self.refresh()

    def refresh(self):
        for i in self.tree.get_children():
            self.tree.delete(i)

        folders = scan_train_safe_Connected_folders()
        for folder in folders:
            rep = analyze_train_safe_Connected_folder(folder)
            if "error" in rep:
                self.tree.insert("", tk.END, values=(folder, "Err", "Err", rep["error"]))
            else:
                self.tree.insert(
                    "",
                    tk.END,
                    values=(
                        f"{folder} ({rep['count']})",
                        str(rep["unique_dims"]),
                        str(rep["health"]),
                        str(rep["edge_health"]),
                    ),
                )

# =========================================================
#  Grid + Features (Rectangular & Irregular) with Grid/Supergrid
# =========================================================

class TrainDataSelectGUI:
    """
    使用 Tkinter 下拉選單選擇 Train_<topology>_* 資料夾，
    並顯示 topology / m,n 範圍 / N / hole_ratio。
    """

    def __init__(self, master, folder_callback):
        self.master = master
        self.master.title("選擇訓練資料資料夾")

        self.folder_callback = folder_callback

        ttk.Label(master, text="請選擇訓練資料資料夾：").pack(pady=5)

        self.folders = scan_train_safe_Connected_folders()

        # 準備顯示文字與實際資料夾的映射
        self.labels = []
        self.label_to_folder = {}

        for folder in self.folders:
            label = self._make_pretty_label(folder)
            self.labels.append(label)
            self.label_to_folder[label] = folder

        self.combo = ttk.Combobox(master, values=self.labels, state="readonly", width=80)
        if self.labels:
            self.combo.current(0)
        self.combo.pack(pady=5)

        btn_frame = ttk.Frame(master)
        btn_frame.pack(pady=10)

        ok_btn = ttk.Button(btn_frame, text="確定", command=self.on_ok)
        ok_btn.pack(side=tk.LEFT, padx=5)

        cancel_btn = ttk.Button(btn_frame, text="取消", command=self.master.destroy)
        cancel_btn.pack(side=tk.LEFT, padx=5)

    def _make_pretty_label(self, folder):
        """
        讀取 meta.json 與資料夾名稱，組合成：
        [topology] m:min-max, n:min-max, N:圖數, H:hole_ratio  (folder)
        """
        topo = "?"
        hole = "?"
        # 從資料夾名稱推測
        parts = folder.split("_")
        for p in parts:
            if p.lower() in ("grid", "supergrid"):
                topo = p
            elif p.startswith("H"):
                hole = p[1:]

        m_min = m_max = n_min = n_max = None
        N = None
        meta_path = os.path.join(str(TRAINPATH), folder, "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                ms = [item.get("m") for item in meta if "m" in item]
                ns = [item.get("n") for item in meta if "n" in item]
                if ms:
                    m_min, m_max = min(ms), max(ms)
                if ns:
                    n_min, n_max = min(ns), max(ns)
                N = len(meta)
            except Exception:
                pass

        def rng(a, b):
            if a is None or b is None:
                return "?"
            return f"{a}-{b}"

        m_str = rng(m_min, m_max)
        n_str = rng(n_min, n_max)
        N_str = str(N) if N is not None else "?"

        label = f"[{topo}] m:{m_str}, n:{n_str}, N:{N_str}, H:{hole}  ({folder})"
        #label = f"[{topo}] : {folder}"
        return label

    def on_ok(self):
        sel_label = self.combo.get()
        if not sel_label:
            messagebox.showerror("錯誤", "請先選擇資料夾")
            return
        folder = self.label_to_folder.get(sel_label)
        if folder is None:
            messagebox.showerror("錯誤", "選擇的資料夾無效")
            return
        self.folder_callback(folder)
        self.master.destroy()


def select_train_folder_gui():
    """
    彈出 GUI 讓使用者選擇訓練資料資料夾。
    回傳選取到的資料夾名稱，若未選擇則回傳 None。
    """
    folders = scan_train_safe_Connected_folders()
    if not folders:
        print(f"❌ 找不到任何 Train_* 資料夾 (TRAINPATH={TRAINPATH})")
        return None

    selected = {"folder": None}

    def cb(folder):
        selected["folder"] = folder

    root = tk.Tk()
    gui = TrainDataSelectGUI(root, cb)
    root.mainloop()
    return os.path.join(str(TRAINPATH), selected["folder"]) if selected["folder"] else None


def select_model_file_gui():
    """彈出檔案選擇視窗，從 MODELPATH (= 2Models/) 選擇一個 *.pt 權重檔。"""
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        initial_dir = str(MODELPATH) if MODELPATH is not None else os.getcwd()
        path = filedialog.askopenfilename(
            title="選擇模型權重檔 (*.pt)",
            initialdir=initial_dir,
            filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")],
        )
        root.destroy()
    except Exception as e:
        print(f"❌ 開啟模型檔選擇視窗失敗: {e}")
        return None

    if not path:
        return None
    try:
        return Path(path)
    except Exception:
        return None

def main_menu():
    print("\n=== Dom-SupergridGrid_oneGNN v38 (v23.2) (oneGNN + testset + beam research- RL enhanced version) ===")
    while True:
        print("\n=========================================================================")
        print(" Dom-SupergridGrid_GNNs- v38 (oneGNN + testset + beam research- RL enhanced version _ GPSGov+Tranformer(Dynamic)+GIN_Nice; But, GCN/GAT/SAGE/GraphTransform(static)_Poor)")
        print(" Irregular Grid / Supergrid Domination GNN Suite + TrainSafe Manager (簡化)")
        print(" No trained dataset:   7 => 8 => 9(10,12,14)")
        print(" Have trained dataset: 8 (9) => 10(12,14)")
        print(" TestSet: 13 => 9 => 14")
        print(" Feature #: 29+5(RL) = 34 dim")
        print("\n=========================================================================")
        print("1.  設定 Grid / Supergrid 模式 (預設Supergrid，可切換為Grid)")
        print("2.  Train SafeData GUI Manager")
        print("3.  一鍵掃描、校正 Train_<topology>_* (報告模式)")
        print("4.  設定 GNN 補點模式 (Beam Search / Guided Greedy / Auto)")
        print("5.  設定 Beam Search 參數")
        print("6.  選擇要執行的 GNN (GCN/GAT/SAGE/GIN/Transformer/GraphTransform/GPSConv)")
        print("7.  產生訓練集資料 (Train_<topology>_*, 不含模型名稱) (Irregular Grid/Supergrid Train_<topology>_*) - mxn + 挖洞比例")
        print("8.  讀取既有訓練集資料並訓練選定的 GNN（訓練集不綁定模型/不指定model檔名）")
        print("9.  選取已訓練過的資料集目錄並載入『選定的』GNN 模型 (不重新訓練)")
        print("10. 使用 1 個已載入的GNN模型測試 (Irregular 測試圖 + 可切換 Beam/GuidedGreedy 補點)")
        print("11. (保留) 一次測試 7 個 GNNs模型並比較（需自行擴充回多模型訓練）")
        print("12. 一次測試 3 個方法 - ILP + Greedy + 1個GNN模型 - 並繪圖比較結果 (Irregular + Beam/Guided completion)")
        print("13. 產生『測試』data set (Irregular 測試圖集合) ")
        print("14. 執行 TestSet：一次跑 GNN + Greedy + ILP 並輸出 CSV")
        print("15. (Check) 實驗模式：多尺寸 + 多挖洞比例，自動比較並輸出 CSV  (可程式中設定– 挖洞不連通多次後即停止挖洞，並輸出挖洞例) ")
        print("16. (Check) 實驗結果視覺化與排名表 (讀取最新 CSV 並依 支配數和挖洞比例為各方法繪製折線圖)")
        print("0.  離開")
        try:
            choice = input("請選擇功能：").strip()
        except EOFError:
            break

        if choice == "1":
            menu_set_topology()
        elif choice == "2":
            print("\n[開啟 TrainSafe Manager GUI] ...")
            root = tk.Tk()
            app = TrainSafeManagerGUI(root)
            root.mainloop()
        elif choice == "3":
            batch_auto_fix_train_safe(
                feature_fix=True,
                edge_fix_mode="shape_then_rebuild"
            )
        elif choice == "4":
            menu_set_gnn_completion_mode()
        elif choice == "5":
            menu_set_beam_params()
        elif choice == "6":
            choose_one_gnn()
        elif choice == "7":
            generate_training_data()
        elif choice == "8":
            train_one_model_from_safe_folder()
        elif choice == "9":
            load_models_from_model_folder()
        elif choice == "10":
            test_current_model()
        elif choice == "11":
            test_all_7_GNNmodels()
        elif choice == "12":
            test_all_3_methods()
        elif choice == "13":
            generate_test_data()
        elif choice == "14":
            run_testset_gnn_greedy_ilp()
        elif choice == "15":
            run_experiment_multi_size_and_holes()
        elif choice == "16":
            analyze_and_plot_experiment_results()
        elif choice == "0":
            print("結束程式。")
            break
        else:
            print("❌ 無效選項，請重新輸入。")

if __name__ == "__main__":
    main_menu()