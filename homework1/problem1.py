def pr1(n):
    m1 = 1
    m2 = 0
    m3 = 1
    for i in range(1, n+1, 2):
        m1 *= i # 計算連續基數相乘值a
        m2 += i # 計算連續基數相加結果b
        m3 += 1/i ## 計算連續基數的倒數相加結果c
    
    return m1, m2, m3

ans_11 = pr1(11)
ans_12 = pr1(21)
ans_13 = pr1(51)
ans_14 = pr1(101)
print(f'The result is {ans_11}.')
print(f'The result is {ans_12}.')
print(f'The result is {ans_13}.')
print(f'The result is {ans_14}.')
