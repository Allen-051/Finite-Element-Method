# 有限元素法期末個人報告的第二題程式碼

# 宣告一個k(3*3)矩陣，共有四個元素所以從k1到k4四個(3*3)矩陣
# 宣告一個u(5*1)向量
# 宣告一個f(5*1)向量
# 宣告一個Q(5*5)矩陣
# k * u = f + Q
# 求解 u
import numpy as np
from scipy.linalg import solve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 設定中文字型
plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號

# 節點座標
nodes = np.array([
    [0, 0],           # 1
    [1, 0],           # 2
    [0.9239, 0.3827], # 3
    [0.7071, 0.7071], # 4
    [0.3827, 0.9239], # 5
    [0, 1]            # 6
])

# 每個元素的節點編號（以0為起點）
elements = [
    [0, 1, 2],  # element 1: 1,2,3
    [0, 2, 3],  # element 2: 1,3,4
    [0, 3, 4],  # element 3: 1,4,5
    [0, 4, 5]   # element 4: 1,5,6
]

# 假設每個元素的剛性矩陣都叫 ke (3x3)，你可以根據實際計算填入
# 這裡用隨機數舉例
        #1,     #2,     #3
k1 = [
    [0.1989, -0.0995, -0.0995], # 1
    [-0.0995, 1.3066, -1.2071], # 2
    [-0.0995, -1.2071, 1.3066]  # 3
    ]

        #1,     #3,     #4
k2 = [
    [0.1989, -0.0995, -0.0995], # 1
    [-0.0995, 1.3066, -1.2071], # 3
    [-0.0995, -1.2071, 1.3066]  # 4
    ]

        #1,     #4,     #5
k3 = [
    [0.1989, -0.0995, -0.0995], # 1 
    [-0.0995, 1.3066, -1.2071], # 4
    [-0.0995, -1.2071, 1.3066]  # 5
    ]

        #1,     #5,     #6
k4 =[
    [0.1989, -0.0995, -0.0995], # 1
    [-0.0995, 1.3066, -1.2071], # 5
    [-0.0995, -1.2071, 1.3066]  # 6
    ]

k_list = [np.array(k1), np.array(k2), np.array(k3), np.array(k4)]

K = np.zeros((6, 6))
for e, elem in enumerate(elements):
    ke = k_list[e]
    for i in range(3):
        for j in range(3):
            K[elem[i], elem[j]] += ke[i][j]
print("Global stiffness matrix K:")
df_K = pd.DataFrame(K)
print(df_K.round(4).to_string(index=False, header=False))

# 計算三角形元素面積的函數
def triangle_area(nodes, element):
    i, j, k = element
    x1, y1 = nodes[i]
    x2, y2 = nodes[j]
    x3, y3 = nodes[k]
    return 0.5 * abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))

# 計算所有元素的面積
areas = [triangle_area(nodes, elem) for elem in elements]
for idx, area in enumerate(areas, 1):
    print(f"Area of element {idx}: {area:.4f}")

# 定義 f0
f0 = 2 * 5

# 計算每個元素的面積
areas = [triangle_area(nodes, elem) for elem in elements]

# 組裝外力向量 f (6x1)
f = np.zeros(6)
for elem_idx, elem in enumerate(elements):
    area = areas[elem_idx]
    for node in elem:
        f[node] += area * f0 / 3  # 每個元素的外力等分到三個節點

f = f.reshape((6, 1))
print("\nf vector:")
df_f = pd.DataFrame(f)
print(df_f.round(4).to_string(index=False, header=False))

# 定義 u 向量 (6x1)，u1 為未知數，其餘為 0
u = np.zeros((6, 1))

# 先求解 u1
# 只用第1列 (index 0) 的方程式：K[0,:] * u = f[0] + Q[0]，Q[0]=0，u[1:]=0
K11 = K[0, 0]
RHS = f[0, 0]  # Q[0]=0
u1 = RHS / K11
u[0, 0] = u1

# 反推 Q2~Q6
Q = np.zeros((6, 1))
Q[0, 0] = 0
for i in range(1, 6):
    Q[i, 0] = (K[i, 0] * u1) - f[i, 0]  # 只考慮 u1 影響，其餘 u=0

# 輸出
print("\nK matrix:")
print(df_K.round(4).to_string(index=False, header=False))
print("\nf vector:")
print(df_f.round(4).to_string(index=False, header=False))
print("\nQ vector:")
df_Q = pd.DataFrame(Q)
print(df_Q.round(4).to_string(index=False, header=False))
print("\nu vector:")
df_u = pd.DataFrame(u)
print(df_u.round(4).to_string(index=False, header=False))

# 解析解 u_exact 計算與輸出
u_exact = []
for x, y in nodes:
    u_val = 5 * 0.5 * (1 - x**2 - y**2)
    u_exact.append([u_val])
u_exact = np.array(u_exact)

print("\nu_exact vector:")
df_ue = pd.DataFrame(u_exact)
print(df_ue.round(4).to_string(index=False, header=False))

rel_error = []
for i in range(len(u_exact)):
    denom = u_exact[i, 0]
    num = u[i, 0]
    if abs(denom) > 1e-12:
        rel_error.append(abs((num - denom) / denom * 100))
    else:
        rel_error.append(np.nan)
rel_error = np.array(rel_error).flatten()

# 合併輸出
result_df = pd.DataFrame({
    "u": u.flatten(),
    "f": f.flatten(),
    "q": Q.flatten(),
    "u_exact": u_exact.flatten(),
    "rel_error(%)": rel_error
})

print("\nu, f, q, u_exact, rel_error(%)：")
print(result_df.to_string(index=False, float_format="%.6f"))

# 劃出元素切割示意圖
plt.figure(figsize=(6, 6))
plt.scatter(nodes[:, 0], nodes[:, 1], c='black', zorder=3)
for i, (x, y) in enumerate(nodes):
    plt.text(x + 0.015, y + 0.015, str(i+1), fontsize=8, color='black')

# 畫出每個元素的三角形邊，並在幾何中心標註元素編號
for idx, e in enumerate(elements):
    n1, n2, n3 = e 
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
