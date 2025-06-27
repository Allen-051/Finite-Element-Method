import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 設定中文字型
plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號

# === 1. 建立完整座標清單，確保元素需要用到的點都存在 ===
nodes = [
    (0.00, 0.00), 
    (0.25, 0.00), 
    (0.50, 0.00), 
    (0.75, 0.00), 
    (1.00, 0.00),
    (0.00, 0.25), 
    (0.25, 0.25), 
    (0.50, 0.25), 
    (0.75, 0.25),
    (0.00, 0.50), 
    (0.25, 0.50), 
    (0.50, 0.50),
    (0.00, 0.75), 
    (0.25, 0.75), 
    (0.00, 1.00),
    (0.309, 0.951), 
    (0.5878, 0.809), 
    (0.809, 0.5878), 
    (0.951, 0.309)
]
nodes = np.array(nodes)

# === 2. 定義23個元素 ===
elements = [
    [1, 2, 6], #1
    [6, 7, 10], #2
    [6, 2, 7], #3
    [2, 3, 7], #4
    [10, 11, 13], #5
    [10, 7, 11], #6
    [7, 8, 11], #7
    [7, 3, 8], #8
    [8, 3, 4], #9
    [15, 13, 14], #10
    [13, 11, 14], #11
    [11, 12, 14], #12
    [11, 8, 12], #13
    [8, 9, 12], #14
    [8, 4, 9], #15
    [9, 4, 5], #16
    [15, 14, 16], #17
    [16, 14, 17], #18
    [14, 12, 17], #19
    [17, 12, 18], #20
    [18, 12, 9], #21
    [18, 9, 19], #22 
    [9, 5, 19]] #23

# === 3. 建立節點 DataFrame ===
node_df = pd.DataFrame(nodes, columns=["x", "y"])
node_df.index += 1
node_df.index.name = "Node"

# === 4. 計算剛度矩陣 ===
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

K_global = np.zeros((19, 19))
areas = []

for e in elements:
    n1, n2, n3 = e
    p1, p2, p3 = nodes[n1-1], nodes[n2-1], nodes[n3-1]
    area = triangle_area(p1, p2, p3)
    areas.append(area)
    beta = compute_betas(p1, p2, p3)
    gamma = compute_gammas(p1, p2, p3)
    k = compute_k_matrix(area, beta, gamma)
    for i_local, i_global in enumerate([n1, n2, n3]):
        for j_local, j_global in enumerate([n1, n2, n3]):
            K_global[i_global-1, j_global-1] += k[i_local, j_local]

K_global_df = pd.DataFrame(K_global)
K_global_df.index += 1
K_global_df.columns = [str(i) for i in range(1, 20)]

# 顯示資料
print("\n節點座標表（補足邊界）:")
print(node_df.to_string())

print("\n總體K矩陣（補足邊界）:")
print(K_global_df.round(6).to_string())

element_data = []
for idx, e in enumerate(elements):
    n1, n2, n3 = e
    p1, p2, p3 = nodes[n1-1], nodes[n2-1], nodes[n3-1]
    area = areas[idx]
    element_data.append([
        idx+1, n1, n2, n3,
        p1[0], p1[1], p2[0], p2[1], p3[0], p3[1],
        area
    ])

element_df = pd.DataFrame(
    element_data,
    columns=["Element", "n1", "n2", "n3", "x1", "y1", "x2", "y2", "x3", "y3", "Area"]
)
print("\n元素與面積（含節點座標）:")
print(element_df.to_string(index=False))


# === 6. 圖形繪製 ===
plt.figure(figsize=(6, 6))
plt.scatter(nodes[:, 0], nodes[:, 1], c='black', zorder=3)
for i, (x, y) in enumerate(nodes):
    plt.text(x + 0.015, y + 0.015, str(i+1), fontsize=8, color='blue')

# 畫出每個元素的三角形邊
for e in elements:
    n1, n2, n3 = [idx - 1 for idx in e]  # 轉成0-based index
    tri_x = [nodes[n1][0], nodes[n2][0], nodes[n3][0], nodes[n1][0]]
    tri_y = [nodes[n1][1], nodes[n2][1], nodes[n3][1], nodes[n1][1]]
    plt.plot(tri_x, tri_y, 'b-', linewidth=0.5, alpha=0.7)

theta = np.linspace(0, np.pi / 2, 100)
x_arc = np.cos(theta)
y_arc = np.sin(theta)
plt.plot(x_arc, y_arc, 'g-', label="Quarter Circle", zorder=1)

plt.plot([0, 1], [0, 0], 'k--', linewidth=1)
plt.plot([0, 0], [0, 1], 'k--', linewidth=1)
plt.gca().set_aspect('equal')
plt.grid(False)
plt.title("節點與元素示意圖（19節點、23元素）")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# 定義forcing term向量
num_nodes = 19
f0 = 10  # f0 = 2 * G * θ = 10
f_vector = np.zeros((num_nodes, 1))

for idx, e in enumerate(elements):
    area = areas[idx]
    contribution = f0 * area / 3
    for node in e:
        f_vector[node - 1] += contribution

f_vector_df = pd.DataFrame(f_vector, columns=["f"])
f_vector_df.index += 1
f_vector_df.index.name = "Node"