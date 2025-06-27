import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import solve
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 設定中文字型
plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號

# === 1. 定義節點與元素 ===
nodes = [
    (0.00, 0.00), (0.25, 0.00), (0.50, 0.00), (0.75, 0.00), (1.00, 0.00),
    (0.00, 0.25), (0.25, 0.25), (0.50, 0.25), (0.75, 0.25),
    (0.00, 0.50), (0.25, 0.50), (0.50, 0.50),
    (0.00, 0.75), (0.25, 0.75), (0.00, 1.00),
    (0.309, 0.951), (0.5878, 0.809), (0.809, 0.5878), (0.951, 0.309)
]
nodes = np.array(nodes)

elements = [
    [1, 2, 6], [6, 7, 10], [6, 2, 7], [2, 3, 7],
    [10, 11, 13], [10, 7, 11], [7, 8, 11], [7, 3, 8],
    [8, 3, 4], [15, 13, 14], [13, 11, 14], [11, 12, 14],
    [11, 8, 12], [8, 9, 12], [8, 4, 9], [9, 4, 5],
    [15, 14, 16], [16, 14, 17], [14, 12, 17], [17, 12, 18],
    [18, 12, 9], [18, 9, 19], [9, 5, 19]
]

# === 2. 面積與剛度矩陣函數 ===
def triangle_area(p1, p2, p3):
    return 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))

def compute_betas(p1, p2, p3):
    return [p2[1]-p3[1], p3[1]-p1[1], p1[1]-p2[1]]

def compute_gammas(p1, p2, p3):
    return [p3[0]-p2[0], p1[0]-p3[0], p2[0]-p1[0]]

def compute_k_matrix(area, beta, gamma):
    k = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            k[i][j] = (beta[i]*beta[j] + gamma[i]*gamma[j]) / (4 * area)
    return k


# === 3. 初始化 K, f, q ===
num_nodes = len(nodes)
K_global = np.zeros((num_nodes, num_nodes))
f_vector = np.zeros((num_nodes, 1))
q_vector = np.zeros((num_nodes, 1))
f0 = 10
areas = []

for e in elements:
    n1, n2, n3 = e
    p1, p2, p3 = nodes[n1-1], nodes[n2-1], nodes[n3-1]
    area = triangle_area(p1, p2, p3)
    areas.append(area)
    beta = compute_betas(p1, p2, p3)
    gamma = compute_gammas(p1, p2, p3)
    k = compute_k_matrix(area, beta, gamma)

    # 組裝 K_global
    for i_local, i_global in enumerate([n1, n2, n3]):
        for j_local, j_global in enumerate([n1, n2, n3]):
            K_global[i_global-1, j_global-1] += k[i_local, j_local]

    # 加入 f 向量（平均分配到3個節點）
    for node in [n1, n2, n3]:
        f_vector[node-1] += f0 * area / 3

# === 4. 定義邊界條件並求解 u ===
boundary_nodes = [5, 15, 16, 17, 18, 19]
interior_nodes = [i for i in range(1, num_nodes+1) if i not in boundary_nodes]
interior_idx = [i-1 for i in interior_nodes]

# 邊界點強制項（可依需求調整值，這裡設為1）
for node in boundary_nodes:
    q_vector[node-1] = 1.0

# 求解 u（僅對內部點）
K_reduced = K_global[np.ix_(interior_idx, interior_idx)]
rhs = f_vector[interior_idx] + q_vector[interior_idx]
u_reduced = solve(K_reduced, rhs)

# 還原完整 u 向量
u_vector = np.zeros((num_nodes, 1))
for i, idx in enumerate(interior_idx):
    u_vector[idx] = u_reduced[i]

# === 5. 加入解析解與誤差 ===
u_exact = []
for x, y in nodes:
    u_val = 5 * 0.5 * (1 - x**2 - y**2)
    u_exact.append(u_val)
u_exact = np.array(u_exact).flatten()

rel_error = []
for i in range(len(u_exact)):
    denom = u_exact[i]
    num = u_vector[i, 0]
    if abs(denom) > 1e-12:
        rel_error.append(abs((num - denom) / denom * 100))
    else:
        rel_error.append(np.nan)
rel_error = np.array(rel_error).flatten()

# === 6. 輸出總結果表格 ===
df_result = pd.DataFrame({
    "x": nodes[:, 0],
    "y": nodes[:, 1],
    "u": u_vector.flatten(),
    "f": f_vector.flatten(),
    "q": q_vector.flatten(),
    "u_exact": u_exact,
    "rel_error(%)": rel_error
})
df_result.index += 1
df_result.index.name = "Node"

# 節點資訊以DataFrame輸出
node_df = pd.DataFrame({
    '節點編號': np.arange(1, len(nodes)+1),
    '節點座標': [f"({x:.4f}, {y:.4f})" for x, y in nodes]
})

# 元素資訊以DataFrame格式輸出
# 構成節點以1-based顯示
el_df = pd.DataFrame({
    '元素編號': np.arange(1, len(elements)+1),
    '構成節點': [f"({e[0]}, {e[1]}, {e[2]})" for e in elements],
    '元素面積': [f"{a:.4f}" for a in areas]
})

# 輸出每個元素的 beta, gamma 參數
betas_list = []
gammas_list = []
for e in elements:
    n1, n2, n3 = e
    p1, p2, p3 = nodes[n1-1], nodes[n2-1], nodes[n3-1]
    beta = [p2[1]-p3[1], p3[1]-p1[1], p1[1]-p2[1]]
    gamma = [p3[0]-p2[0], p1[0]-p3[0], p2[0]-p1[0]]
    betas_list.append(beta)
    gammas_list.append(gamma)

beta_gamma_df = pd.DataFrame({
    '元素編號': np.arange(1, len(elements)+1),
    'beta1': [b[0] for b in betas_list],
    'beta2': [b[1] for b in betas_list],
    'beta3': [b[2] for b in betas_list],
    'gamma1': [g[0] for g in gammas_list],
    'gamma2': [g[1] for g in gammas_list],
    'gamma3': [g[2] for g in gammas_list]
})
# === 7. 輸出到終端機 ===
print("\n節點解與向量彙總（u, f, q, u_exact, rel_error）:")
print(df_result.round(6).to_string())

print("\n節點資訊：")
print(node_df.to_string(index=False))

print("\n元素資訊：")
print(el_df.to_string(index=False))

# 輸出元素 beta, gamma 參數
print("\n元素 beta, gamma 參數：")
print(beta_gamma_df.round(6).to_string(index=False))

# K_global 矩陣以 DataFrame 輸出
print("\n整體勁度矩陣 K_global：")
df_K = pd.DataFrame(K_global)
print(df_K.round(4).to_string(index=False, header=False))

# 輸出 f 向量
print("\nf 向量：")
df_f = pd.DataFrame(f_vector, columns=["f"])
df_f.index += 1
df_f.index.name = "Node"
print(df_f.round(6).to_string())

# 匯出所有表格到同一份CSV
with open('23_element_data.csv', 'w', encoding='utf-8-sig') as f:
    f.write('節點資訊：\n')
    node_df.to_csv(f, index=False)
    f.write('\n元素資訊：\n')
    el_df.to_csv(f, index=False)
    f.write('\n元素 beta, gamma 參數：\n')
    beta_gamma_df.to_csv(f, index=False)
    f.write('\nf 向量：\n')
    df_f.to_csv(f)
    f.write('\n整體勁度矩陣 K_global：\n')
    df_K.to_csv(f, index=False, header=False)
    f.write('\n節點解與向量彙總（u, f, q, u_exact, rel_error）：\n')
    df_result.to_csv(f)

# 劃出元素切割示意圖
plt.figure(figsize=(6, 6))
plt.scatter(nodes[:, 0], nodes[:, 1], c='black', zorder=3)
for i, (x, y) in enumerate(nodes):
    plt.text(x + 0.015, y + 0.015, str(i+1), fontsize=8, color='black')

# 畫出每個元素的三角形邊，並在幾何中心標註元素編號
for idx, e in enumerate(elements):
    n1, n2, n3 = [idx - 1 for idx in e]  # 轉成0-based index
    tri_x = [nodes[n1][0], nodes[n2][0], nodes[n3][0], nodes[n1][0]]
    tri_y = [nodes[n1][1], nodes[n2][1], nodes[n3][1], nodes[n1][1]]
    plt.plot(tri_x, tri_y, 'b-', linewidth=0.5, alpha=0.7)
    # 計算三個節點的幾何中心
    cx = (nodes[n1][0] + nodes[n2][0] + nodes[n3][0]) / 3
    cy = (nodes[n1][1] + nodes[n2][1] + nodes[n3][1]) / 3
    plt.text(cx, cy, str(idx+1), fontsize=10, color='purple', ha='center', va='center', fontweight='bold')

theta = np.linspace(0, np.pi / 2, 100)
x_arc = np.cos(theta)
y_arc = np.sin(theta)
plt.plot(x_arc, y_arc, 'g-', label="Quarter Circle", zorder=1)

plt.plot([0, 1], [0, 0], 'k--', linewidth=1)
plt.plot([0, 0], [0, 1], 'k--', linewidth=1)
plt.gca().set_aspect('equal')
plt.grid(False)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
