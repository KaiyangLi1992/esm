import numpy as np

# 创建示例的布尔数组
matrix = np.array([[False, True, False],
                   [True, False, True],
                   [True, False, False],
                   [False, False, False],
                   [True, True, False]])

# 要排除的行的索引列表
exclude_indices = [0, 2, 3]  # 假设要排除 [x, y, z]

# 创建一个布尔掩码来排除指定的行
exclude_mask = np.ones(matrix.shape[0], dtype=bool)
exclude_mask[exclude_indices] = False

# 根据掩码过滤剩余的行
filtered_matrix = matrix[exclude_mask]

# 计算每一行中True值的数量
row_true_counts = np.sum(filtered_matrix, axis=1)

# 找到True值最少的行的索引
min_true_row_index_filtered = np.argmin(row_true_counts)

# 转换为原始矩阵中的索引
min_true_row_index_original = np.where(exclude_mask)[0][min_true_row_index_filtered]

# 输出结果
print("排除了 [x, y, z] 后，原始矩阵中True值最少的行索引:", min_true_row_index_original)
