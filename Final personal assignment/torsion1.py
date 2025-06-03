# 有限元素法期末個人報告的第二題程式碼

# 宣告一個k(3*3)矩陣，共有四個元素所以從k1到k4四個(3*3)矩陣
# 宣告一個u(5*1)向量
# 宣告一個f(5*1)向量
# 宣告一個Q(5*5)矩陣
# k * u = f + Q
# 求解 u
import numpy as np
from scipy.linalg import solve

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
              [ 0,    k2[1,0],  4, -1,  0],
              [ 0,  0, -1,  4, -1],
              [ 0,  0,  0, -1,  4]])
# Define the force vector f (5x1)
f = np.array([[0], [0], [0], [0], [10]])  # Example force vector
# Define the displacement vector u (6x1)
u = np.zeros(5)  # Initial guess for displacements
# Define the Q matrix (5x1)
Q = np.zeros(5)  # Assuming no additional forces or constraints
