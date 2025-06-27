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
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 設定中文字型
plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號

# the coordinates of each nodes
nodes = np.array([
    [0, 0],
    [1, 0],
    [0.7071, 0.7071],
    [0, 1],
    [0.4, 0.4]
])
# find the area of each element
# nodes: 0=node1, 1=node2, 2=node3, 3=node4, 4=node5
# element 1: 節點 1, 2, 5
area1 = 0.5 * abs(
    (nodes[1,0] - nodes[0,0]) * (nodes[4,1] - nodes[0,1]) -
    (nodes[4,0] - nodes[0,0]) * (nodes[1,1] - nodes[0,1])
)

# element 2: 節點 2, 3, 5
area2 = 0.5 * abs(
    (nodes[2,0] - nodes[1,0]) * (nodes[4,1] - nodes[1,1]) -
    (nodes[4,0] - nodes[1,0]) * (nodes[2,1] - nodes[1,1])
)

# element 3: 節點 3, 4, 5
area3 = 0.5 * abs(
    (nodes[3,0] - nodes[2,0]) * (nodes[4,1] - nodes[2,1]) -
    (nodes[4,0] - nodes[2,0]) * (nodes[3,1] - nodes[2,1])
)

# element 4: 節點 4, 1, 5
area4 = 0.5 * abs(
    (nodes[0,0] - nodes[3,0]) * (nodes[4,1] - nodes[3,1]) -
    (nodes[4,0] - nodes[3,0]) * (nodes[0,1] - nodes[3,1])
)

print(f'A1(n1、n2、n5):{area1}')
print(f'A2(n2、n3、n5):{area2}')
print(f'A3(n3、n4、n5):{area3}')
print(f'A4(n4、n1、n5):{area4}')

# element 1


k1 = np.array([[ 0.65, 0.1,  -0.75],
              [ 0.1,  0.4, -0.5],
              [ -0.75, -0.5,  1.25]])
k2 = np.array([[ 0.371, -0.1,  -0.2071],
              [ -0.1, 0.8466, -0.7466],
              [ -0.2071, -0.7466,  0.9537]])
k3 = np.array([[ 0.9537, -0.7466,  -0.2071],
              [ -0.7466,  0.8466, 0.1],
              [ -0.2071, 0.1,  0.3071]])
k4 = np.array([[ 0.65, -0.75,  0.1],
              [ -0.75,  1.25, -0.5],
              [ 0.1, -0.5,  0.4]])
# Combine k1, k2, k3, k4 into a 5x5 matrix
k  =  np.array([[ k1[0,0] + k4[0,0],     k1[1,0],     0,    k4[0,2],     k1[0,2] + k4[0,1]],
              [k1[1,0],     k1[1,1] + k2[1,0],    k2[1,0],        0,     k1[1,2] + k2[0,2]],
              [    0,    k2[1,0],     k2[1,1] + k3[1,1],    k3[1,2],     k2[1,2] + k3[1,0]],
              [ k4[2,0],      0,      k3[2,1],    k3[2,2] + k4[2,2],     k3[2,0] + k4[1,2]],
              [ k1[0,2]+k4[0,1],k1[2,1]+k2[2,0],k3[2,1]+k3[0,1],k3[0,2]+k4[1,2],k1[2,2]+k2[2,2]+k3[0,0]+k4[1,1]]])

f0 = 2 * 5 
f1 = area1*f0/3
f2 = area2*f0/3
f3 = area3*f0/3
f4 = area4*f0/3

# Define the force vector f (5x1)
f = np.array([[f1+f4], [f1+f2], [f2+f3], [f3+f4], [f1+f2+f3+f4]]) 
# Define the displacement vector u (5x1)
# u = np.array([[u1],[0],[0],[0],[u5]]) 

# 先解 u1, u5
# 取 k 的第1列和第5列（index 0, 4），只保留 u1, u5 相關的係數
k_reduced = np.array([
    [k[0,0], k[0,4]],
    [k[4,0], k[4,4]]
])
f_reduced = np.array([
    f[0,0],
    f[4,0]
])
# Q1, Q5 = 0
rhs = f_reduced  # 因為 Q1, Q5 = 0

# 解 u1, u5
u1_u5 = np.linalg.solve(k_reduced, rhs)
u1 = u1_u5[0]
u5 = u1_u5[1]
#print(f"u1 = {u1:.4f}, u5 = {u5:.4f}")

# 再解 Q2, Q3, Q4
Q2 = k[1,0]*u1 + k[1,4]*u5 - f[1,0]
Q3 = k[2,0]*u1 + k[2,4]*u5 - f[2,0]
Q4 = k[3,0]*u1 + k[3,4]*u5 - f[3,0]
#print(f"Q2 = {Q2:.4f}, Q3 = {Q3:.4f}, Q4 = {Q4:.4f}")

Q = np.array([[0],[Q2],[Q3],[Q4],[0]])
# ****結果輸出****
# 輸出 K 矩陣
print("K matrix:")
df_k = pd.DataFrame(k)
print(df_k.round(4).to_string(index=False, header=False))

# 輸出 u 向量 (5x1)
print("\nu vector:")
u_vec = np.array([[u1], [0], [0], [0], [u5]])
df_u = pd.DataFrame(u_vec)
print(df_u.round(4).to_string(index=False, header=False))

# 輸出 f 向量 (5x1)
print("\nf vector:")
df_f = pd.DataFrame(f)
print(df_f.round(4).to_string(index=False, header=False))

# 輸出 Q 向量 (5x1)
print("\nQ vector:")
Q_vec = np.array([[0], [Q2], [Q3], [Q4], [0]])
df_Q = pd.DataFrame(Q_vec)
print(df_Q.round(4).to_string(index=False, header=False))

# 解析解 u_exact 計算與輸出
u_exact = []
for x, y in nodes:
    u_val = 5 * 0.5 * (1 - x**2 - y**2)
    u_exact.append([u_val])
u_exact = np.array(u_exact)

print("\nu_exact vector:")
df_ue = pd.DataFrame(u_exact)
print(df_ue.round(4).to_string(index=False, header=False))

# 計算相對誤差
rel_error = []
for i in range(len(u_exact)):
    denom = u_exact[i, 0]
    num = u_vec[i, 0]
    if abs(denom) > 1e-12:
        rel_error.append(abs((num - denom) / denom * 100))
    else:
        rel_error.append(np.nan)
rel_error = np.array(rel_error)

# 用 DataFrame 合併輸出
result_df = pd.DataFrame({
    "u": u_vec.flatten(),
    "f": f.flatten(),
    "q": Q_vec.flatten(),
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

# 畫出每個元素的三角形邊
# 元素定義：[[0,1,4],[1,2,4],[2,3,4],[3,0,4]]
elements = [
    [0, 1, 4],  # 1,2,5
    [1, 2, 4],  # 2,3,5
    [2, 3, 4],  # 3,4,5
    [3, 0, 4]   # 4,1,5
]
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

# 畫四分之一圓弧
import numpy as np
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