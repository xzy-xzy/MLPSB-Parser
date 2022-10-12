import torch as tch

def BFS(f, l):
    son, vis = [], [False] * l
    for i in range(l):
        son.append([])
    for i in range(l - 1):
        son[f[i]].append(i)
    q, h, t = [l - 1], 0, 0
    while h <= t:
        x = q[h]
        h += 1
        vis[x] = True
        for y in son[x]:
            if vis[y]:
                return False
            else:
                t += 1
                q.append(y)
    return h == l

def tree_check(W_arc, L):
    _, fa = tch.max(W_arc, dim=2)   # (B, N)
    cnt, bad = 0, [ ]
    for f, l in zip(fa, L):
        if not BFS(f, l):
            bad.append(cnt)
        cnt += 1
    return fa, bad
