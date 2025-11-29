import networkx as nx

# 模拟图
G = nx.Graph()
G.add_edge(0, 3, weight=3)
G.add_edge(0, 4, weight=4)
G.add_edge(0, 1, weight=3)
G.add_edge(3, 2, weight=1)
G.add_edge(4, 2, weight=2)
G.add_edge(1, 4, weight=4)

q = [0, 2]

# 提取的结果
solution = [0, 3, 2]
weight = 4

print("=== 验证路径 ===")
print(f"提取的路径: {solution}")
print(f"起点: {solution[0]}, 终点: {solution[-1]}")
print(f"期望起点: {q[0]}, 期望终点: {q[1]}")

# 检查路径连续性
path_valid = True
actual_weight = 0
for i in range(len(solution) - 1):
    if G.has_edge(solution[i], solution[i+1]):
        edge_weight = G[solution[i]][solution[i+1]]['weight']
        actual_weight += edge_weight
        print(f"边 {solution[i]} -> {solution[i+1]}: 权重 {edge_weight}")
    else:
        path_valid = False
        print(f"边 {solution[i]} -> {solution[i+1]}: 不存在！")
        break

print(f"\n路径是否有效: {path_valid}")
print(f"实际权重: {actual_weight}")
print(f"GPT给出的权重: {weight}")

# 计算正确的最短路径
shortest_path = nx.shortest_path(G, q[0], q[1], weight='weight')
shortest_weight = nx.shortest_path_length(G, q[0], q[1], weight='weight')
print(f"\n正确的最短路径: {shortest_path}")
print(f"正确的最短权重: {shortest_weight}")

# 准确率
acc1 = 1 if (solution[0] == q[0] and solution[-1] == q[1] and path_valid and actual_weight == shortest_weight) else 0
acc2 = 1 if weight == actual_weight else 0

print(f"\nAcc (路径准确率): {acc1}")
print(f"Acc2 (权重准确率): {acc2}")
