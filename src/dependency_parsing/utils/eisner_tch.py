import torch as tch
def eisner_dp_tch(W, L, DV):

    B, n = W.size(0), W.size(1)

    tri = tch.zeros((B, n, n, 2)).to(DV)
    trap = tch.zeros((B, n, n, 2)).to(DV)
    tri_path = tch.zeros((B, n, n, 2), dtype=int).to(DV)
    trap_path = tch.zeros((B, n, n, 2), dtype=int).to(DV)


    for i in range(1, n):
        for l in range(n - i):

            r = l + i

            # trap[l][r][0] = max{tri[l][k][1] + tri[k+1][r][0]} + W[l][r] (l<=k<r)
            # trap[l][r][1] = max{tri[l][k][1] + tri[k+1][r][0]} + W[r][l] (l<=k<r)

            S = tri[:, l, l:r, 1] + tri[:, l+1:r+1, r, 0]
            score, pos = tch.max(S, dim=1)
            trap[:, l, r, 0], trap_path[:, l, r, 0] = score + W[:, l, r], pos + l
            trap[:, l, r, 1], trap_path[:, l, r, 1] = score + W[:, r, l], pos + l

            # tri[l][r][0] = max{tri[l][k][0] + trap[k][r][0]} (l<=k<r)

            S = tri[:, l, l:r, 0] + trap[:, l:r, r, 0]
            score, pos = tch.max(S, dim=1)
            tri[:, l, r, 0] = score
            tri_path[:, l, r, 0] = pos + l

            # tri[l][r][1] = max{trap[l][k][1] + tri[k][r][1]} (l<k<=r)

            S = trap[:, l, l+1:r+1, 1] + tri[:, l+1:r+1, r, 1]
            score, pos = tch.max(S, dim=1)
            tri[:, l, r, 1] = score
            tri_path[:, l, r, 1] = pos + (l+1)

    head = tch.zeros((B, n), dtype=int)
    for i in range(B):
        dfs(head[i], 0, L[i] - 1, 0, 1, tri_path[i].cpu( ), trap_path[i].cpu( ))
    return head

def dfs(head, l, r, is_rd, is_tri, ptri, ptrap):

    if l == r: return

    if is_tri:
        k = ptri[l, r, is_rd]
        if is_rd:
            dfs(head, l, k, 1, 0, ptri, ptrap)
            dfs(head, k, r, 1, 1, ptri, ptrap)
        else:
            dfs(head, l, k, 0, 1, ptri, ptrap)
            dfs(head, k, r, 0, 0, ptri, ptrap)

    else:
        k = ptrap[l, r, is_rd]
        if is_rd: head[r] = l
        else: head[l] = r
        dfs(head, l, k, 1, 1, ptri, ptrap)
        dfs(head, k+1, r, 0, 1, ptri, ptrap)
