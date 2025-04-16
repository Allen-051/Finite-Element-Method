# 匯總完整程式碼：Rayleigh-Ritz 方法解 N=1,2,3 次近似解並比較 w, w', M, V
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# 1. 符號設定
x, L, q0, EI = sp.symbols('x L q0 EI', real=True, positive=True)
pi = sp.pi

# 2. 定義試函數與荷載函數
phi = [x**2, x**3, x**4]
q_expr = q0 * sp.sin(pi * x / L)
M0 = q0 * L**2 / pi

# 3. Rayleigh-Ritz 通用 K, F 計算函數
def build_K_F(phis):
    n = len(phis)
    K = sp.zeros(n)
    F = sp.zeros(n, 1)
    for i in range(n):
        for j in range(n):
            K[i, j] = sp.integrate(EI * sp.diff(phis[i], x, 2) * sp.diff(phis[j], x, 2), (x, 0, L))
        F[i] = sp.integrate(phis[i] * q_expr, (x, 0, L)) - M0 * sp.diff(phis[i], x).subs(x, L)
    return K, F

# 4. 求解 N=1,2,3
C1, C2, C3 = sp.symbols('C1 C2 C3')
sols = []
for N in [1, 2, 3]:
    phis = phi[:N]
    K, F = build_K_F(phis)
    C_syms = [C1, C2, C3][:N]
    sol = sp.solve(K * sp.Matrix(C_syms) - F, C_syms)
    sols.append(sol)

# 5. 定義近似位移函數 w1, w2, w3
w_exprs = []
for N, sol in zip([1, 2, 3], sols):
    w_expr = sum(sol[C] * phi[i] for i, C in enumerate([C1, C2, C3][:N]))
    w_exprs.append(w_expr)

# 6. 定義理論解（w_exact）
w_exact_expr = - ((x / pi**4) * sp.sin(pi * x / L)
                  - (1 / 6) * (L / pi) * x**3
                  + (1 / 2) * (L**2 / pi - L**2 / pi**4) * x**2
                  - (L / pi)**3 * x)

# 7. 導數函數
def get_derivatives(w_expr):
    wp = sp.diff(w_expr, x)
    M = EI * sp.diff(w_expr, x, 2)
    V = EI * sp.diff(w_expr, x, 3)
    return w_expr, wp, M, V

# 8. 數值化（代入 q0=L=EI=1）
subs_dict = {q0: 1, L: 1, EI: 1}
x_vals = np.linspace(0, 1, 300)

# 所有解含理論解
all_exprs = w_exprs + [w_exact_expr]
labels = [r'$w_1$', r'$w_2$', r'$w_3$', r'$w_{exact}$']
styles = ['--', '-.', ':', '-']

# 整理所有量：w, w', M, V
results = {'w': [], "w'": [], 'M': [], 'V': []}
for expr in all_exprs:
    w, wp, M, V = get_derivatives(expr.subs(subs_dict))
    for key, val in zip(results.keys(), [w, wp, M, V]):
        func = sp.lambdify(x, val, 'numpy')
        results[key].append(func(x_vals))

# 9. 繪圖
titles = ['Deflection w(x)', "Slope w'(x)", 'Bending Moment M(x)', 'Shear Force V(x)']
ylabels = ['w(x)', "w'(x)", 'M(x)', 'V(x)']

for i, key in enumerate(results.keys()):
    plt.figure(figsize=(8, 5))
    for y_vals, label, style in zip(results[key], labels, styles):
        plt.plot(x_vals, y_vals, style, label=label, linewidth=2)
    plt.title(titles[i])
    plt.xlabel('x')
    plt.ylabel(ylabels[i])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
