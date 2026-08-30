# ==============================================================================
# 專題實作：超網格圖最小支配集（Supergrid Graph MDS）
# 1. 自動建立 N x M 網格與正確的 4-方向鄰居（上下左右）
# 2. 執行貪心法（Greedy Algorithm）尋找守衛最佳配置
# 3. 在終端機印出棋盤視覺化文字圖
# ==============================================================================

def create_grid_graph(rows, cols):
    """
    自動生成 N x M 的網格地圖與鄰居關係表
    """
    neighbors = {}
    all_nodes = list(range(1, rows * cols + 1))
    
    for r in range(rows):
        for c in range(cols):
            # 計算當前格子的編號 (1~N)
            node_id = r * cols + c + 1
            neighbors[node_id] = []
            
            # 定義 4-方向鄰居 (上, 下, 左, 右)
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                # 確保鄰居座標沒有超出地圖邊界
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbor_id = nr * cols + nc + 1
                    neighbors[node_id].append(neighbor_id)
                    
    return all_nodes, neighbors


def run_greedy_guards(all_nodes, neighbors):
    """
    貪心演算法：每次選擇能「新保護最多格子」的位置放守衛
    """
    uncovered = set(all_nodes)
    selected_guards = []
    step = 1
    
    while uncovered:
        best_node = None
        best_new_coverage = set()
        
        # 逐一評估每一個格子作為守衛點的效益
        for node in all_nodes:
            coverage = {node}.union(neighbors[node])
            new_coverage = coverage.intersection(uncovered)
            
            if len(new_coverage) > len(best_new_coverage):
                best_new_coverage = new_coverage
                best_node = node
        
        selected_guards.append(best_node)
        uncovered -= best_new_coverage
        
        print(f"【第 {step} 步】在點 {best_node} 放守衛！")
        print(f"  └─ 新保護了：{sorted(list(best_new_coverage))}")
        print(f"  └─ 剩餘未保護格子數：{len(uncovered)}\n")
        step += 1
        
    return selected_guards


def print_visual_grid(rows, cols, guards):
    """
    在終端機中印出網格視覺化圖案
    [G] 代表守衛 (Guard)
    [.] 代表一般被保護的格子
    """
    print("=" * 40)
    print("=== 網格守衛配置視覺化圖 ===")
    print("=" * 40 + "\n")
    guard_set = set(guards)
    
    for r in range(rows):
        row_str = ""
        for c in range(cols):
            node_id = r * cols + c + 1
            if node_id in guard_set:
                row_str += f"[ 🛡️ G{node_id} ] "  # 顯示守衛與格子編號
            else:
                row_str += f"[  .{node_id:2d}  ] "  # 一般格子與編號
        print(row_str)
        print()


# ==============================================================================
# 主程式執行區塊
# ==============================================================================
if __name__ == "__main__":
    ROWS, COLS = 3, 3
    print(f"=== 建立 {ROWS}x{COLS} 網格地圖 ===")
    all_nodes, neighbors = create_grid_graph(ROWS, COLS)
    
    print("\n" + "=" * 40)
    print("=== 開始執行貪心法守衛配置 ===")
    print("=" * 40 + "\n")
    
    guards = run_greedy_guards(all_nodes, neighbors)
    
    print("=" * 40)
    print(f"全區保護完成！總共使用了 {len(guards)} 個守衛。")
    print(f"守衛最佳擺放位置為：{guards}")
    
    # 印出文字版視覺化棋盤
    print_visual_grid(ROWS, COLS, guards)