import numpy as np

def eisner_dp(W, L):

    B, n = W.size(0), W.size(1)
    W = np.array(W)

    tri = np.zeros((B, n, n, 2))
    trap = np.zeros((B, n, n, 2))
    tri_path = np.zeros((B, n, n, 2), dtype=int)
    trap_path = np.zeros((B, n, n, 2), dtype=int)


    for i in range(1, n):
        for l in range(n - i):

            r = l + i

            # trap[l][r][0] = max{tri[l][k][1] + tri[k+1][r][0]} + W[l][r] (l<=k<r)
            # trap[l][r][1] = max{tri[l][k][1] + tri[k+1][r][0]} + W[r][l] (l<=k<r)

            S = tri[:, l, l:r, 1] + tri[:, l+1:r+1, r, 0]
            score, pos = np.max(S, axis=1), np.argmax(S, axis=1) + l
            trap[:, l, r, 0], trap_path[:, l, r, 0] = score + W[:, l, r], pos
            trap[:, l, r, 1], trap_path[:, l, r, 1] = score + W[:, r, l], pos

            # tri[l][r][0] = max{tri[l][k][0] + trap[k][r][0]} (l<=k<r)

            S = tri[:, l, l:r, 0] + trap[:, l:r, r, 0]
            tri[:, l, r, 0] = np.max(S, axis=1)
            tri_path[:, l, r, 0] = np.argmax(S, axis=1) + l

            # tri[l][r][1] = max{trap[l][k][1] + tri[k][r][1]} (l<k<=r)

            S = trap[:, l, l+1:r+1, 1] + tri[:, l+1:r+1, r, 1]
            tri[:, l, r, 1] = np.max(S, axis=1)
            tri_path[:, l, r, 1] = np.argmax(S, axis=1) + (l+1)

    head = np.zeros((B, n), dtype=int)
    for i in range(B):
        dfs(head[i], 0, L[i] - 1, 0, 1, tri_path[i], trap_path[i])
    return head

def dfs(head, l, r, is_rd, is_tri, ptri, ptrap):

    if l == r : return

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
