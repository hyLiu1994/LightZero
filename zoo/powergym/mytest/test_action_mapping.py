def map_to_discrete(value):
    # 第一步：归一化，将 [-1, 1] 映射到 [0, 1]
    normalized = (value + 1) / 2

    # 第二步：缩放，将 [0, 1] 映射到 [0, 32]
    scaled = normalized * 32

    # 将结果转换为整数
    discrete_value = int(round(scaled))

    # 确保结果在 0 到 32 的范围内
    discrete_value = min(max(discrete_value, 0), 32)

    return discrete_value


# 测试示例
test_values = [-1, -0.6213, 0, 0.5, 1]
discrete_values = [map_to_discrete(val) for val in test_values]

print(discrete_values)

if __name__ == '__main__':
    pass