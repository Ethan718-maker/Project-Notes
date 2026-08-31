# ==============================================================================
# 專題實作第二日：帶有障礙物/挖洞的超網格圖 (Supergrid Graph with Holes)
# 1. 自動建構 N x M 超網格圖，自動將障礙物從可站立與需保護清單中排除
# 2. 執行貪心演算法（Greedy Dominating Set）尋找最佳守衛位置
# 3. 終端機視覺化印出地圖（❌ 代表牆壁/洞）
# ==============================================================================

def create_supergrid_graph(rows, cols, obstacles=[]):
    """
    自動生成 N x M 超網格圖，支援設定障礙物/挖洞 (Obstacles/Holes)
    """
    neighbors = {}
    obstacle_set = set(obstacles)
    
    # 所有有效的節點 (自動排除障礙物/洞)
    all_nodes = [node for node in range(1, rows * cols + 1) if node not in obstacle_set]
    
    for r in range(rows):
        for c in range(cols):
            node_id = r * cols + c + 1
            
            # 如果這個格子是障礙物，直接跳過不計算
            if node_id in obstacle_set:
                continue
                
            neighbors[node_id] = []
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbor_id = nr * cols + nc + 1
                    # 只有當鄰居「不是障礙物」時，才加入鄰居清單
                    if neighbor_id not in obstacle_set:
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


def print_visual_grid(rows, cols, guards, obstacles=[]):
    """
    印出帶有障礙物的網格圖
    [ 🛡️ G ] = 守衛
    [  ❌  ] = 障礙物/牆壁/洞 (Hole)
    [  .   ] = 一般被保護的房間
    """
    print("=" * 45)
    print("=== 帶障礙物/挖洞之超網格圖守衛視覺化 ===")
    print("=" * 45 + "\n")
    
    guard_set = set(guards)
    obstacle_set = set(obstacles)
    
    for r in range(rows):
        row_str = ""
        for c in range(cols):
            node_id = r * cols + c + 1
            if node_id in obstacle_set:
                row_str += "[  ❌XX ] "  # 障礙物/洞
            elif node_id in guard_set:
                row_str += f"[ 🛡️ G{node_id:2d} ] "  # 守衛
            else:
                row_str += f"[  .{node_id:2d}  ] "  # 一般被保護房間
        print(row_str)
        print()


# ==============================================================================
# 主程式執行區塊
# ==============================================================================
if __name__ == "__main__":
    ROWS, COLS = 5, 5
    # 設定陷阱牆壁
    HOLES = [2, 4, 10, 12, 14, 22, 24]
    
    print(f"=== 建立 {ROWS}x{COLS} 陷阱超網格圖（牆壁編號：{HOLES}）===")
    all_nodes, neighbors = create_supergrid_graph(ROWS, COLS, obstacles=HOLES)
    
    print("\n" + "=" * 45)
    print("=== 開始執行貪心法守衛配置 ===")
    print("=" * 45 + "\n")
    
    guards = run_greedy_guards(all_nodes, neighbors)
    
    print("=" * 45)
    print(f"全區保護完成！總共使用了 {len(guards)} 個守衛。")
    print(f"守衛最佳擺放位置為：{guards}")
    
    # 印出視覺化圖案
    print_visual_grid(ROWS, COLS, guards, obstacles=HOLES)