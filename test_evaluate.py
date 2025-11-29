import re

ans = "to find the shortest path from node 0 to node 2, let's evaluate all possible paths:\n\n### paths from node 0 to node 2:\n1. **path 0 → 3 → 2**:  \n   weight = 3 (edge 0 → 3) + 1 (edge 3 → 2) = **4**.\n\n2. **path 0 → 4 → 2**:  \n   weight = 4 (edge 0 → 4) + 2 (edge 4 → 2) = **6**.\n\n3. **path 0 → 1 → 4 → 2**:  \n   weight = 3 (edge 0 → 1) + 4 (edge 1 → 4) + 2 (edge 4 → 2) = **9**.\n\n### shortest path:\nthe shortest path is **0 → 3 → 2** with a total weight of **4**.\n\n### final answer:\nthe shortest path from node 0 to node 2 is **[0, 3, 2]** with total length **4**."

q = [0, 2]
entity = "node"

print("=== 测试原始逻辑 ===")
mode_str = "the shortest path from "+entity+' '+str(q[0])+" to "+ entity + ' ' + str(q[1])
print(f"mode_str: {mode_str}")
pos = ans.rfind(mode_str)
print(f"pos: {pos}")

if pos != -1:
    pos = pos + len(mode_str) + 1
    print(f"开始位置: {pos}")
    print(f"从此位置开始的内容: {repr(ans[pos:pos+50])}")
    
    end_pos = len(ans)
    with_pos = ans.find(" with", pos)
    print(f"with_pos: {with_pos}")
    if with_pos != -1:
        end_pos = with_pos
    
    print(f"end_pos: {end_pos}")
    print(f"提取范围的内容: {repr(ans[pos:end_pos])}")
    
    num, flag = 0, 0
    solution = []
    
    for i in range(pos, end_pos):
        if ans[i] >= '0' and ans[i] <= '9':
            num = num * 10 + int(ans[i])
            flag = 1
        else:
            if flag == 1:
                solution.append(num)
                flag = 0
            num = 0
    
    if flag == 1:
        solution.append(num)
    
    print(f"提取的路径: {solution}")

print("\n=== 测试权重提取 ===")
pos = ans.rfind("total length")
print(f"total length pos: {pos}")
if pos == -1:
    pos = ans.rfind("total weight")
    print(f"total weight pos: {pos}")

if pos != -1:
    print(f"从此位置开始: {repr(ans[pos:pos+30])}")
    i = pos
    while i < len(ans) and not (ans[i] >= '0' and ans[i] <= '9'):
        i += 1
    num = 0
    while i < len(ans) and ans[i] >= '0' and ans[i] <= '9':
        num = num * 10 + int(ans[i])
        i += 1
    print(f"提取的权重: {num}")
