#  problem 4.2 用if取出1到100中不是2、3、5、7倍數的數字
result = []
for i in range(1, 101):
    if i % 2 != 0 and i % 3 != 0 and i % 5 != 0 and i % 7 != 0:
        result.append(i)

print(', '.join(map(str, result)))