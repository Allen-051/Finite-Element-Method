# 計算等比級數和
def geomatric_sum(a, b):
    if a == 1:
        return b
    else:
        return (1-a**b) / (1-a) # 等比級數總和公式
# 以陣列計算題目數值   
r_table = [0.1, 0.25, 0.66, 0.99]
m_table = [3, 10, 50, 200, 400]
for r in r_table:
    add = [1 / (1-r)]
    print(add) # 1/ (1-r)的值

for r in r_table:
    ans_array = [geomatric_sum(r,m) for m in m_table + add]
    print(', '.join(f'{ans2:.6f}' for ans2 in ans_array))

