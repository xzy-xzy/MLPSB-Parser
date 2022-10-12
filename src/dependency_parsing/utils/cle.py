import numpy as np
INF = 1e9


class edge:
    def __init__(self, x, y, w):
        self.x = x
        self.y = y
        self.w = w
        self.rx = x
        self.ry = y
        self.select_time = -1

    def get(self):
        return self.x, self.y, self.w

    def get_r(self):
        return self.rx, self.ry, self.select_time


def cle(E, n, m, root, head):

    org_n = n
    sum = 0
    time = 0

    while True:

        fa = [-1] * n
        bst = [-INF] * n

        for i in range(m):
            x, y, w = E[i].get( )
            if x != y and w > bst[y]:
                bst[y] = w
                fa[y] = x
                E[i].select_time = time

        for i in range(n):
            if i != root and bst[i] == -INF:
                return False

        cnt = 0
        bst[root] = 0
        vis = [-1] * n
        cir = [-1] * n

        for i in range(n):
            sum += bst[i]
            x = i
            while x != root and cir[x] == -1 and vis[x] != i:
                vis[x] = i
                x = fa[x]
            if x != root and cir[x] == -1:
                while cir[x] == -1:
                    cir[x] = cnt
                    x = fa[x]
                cnt += 1

        if cnt == 0:
            break

        for i in range(n):
            if cir[i] == -1:
                cir[i] = cnt
                cnt += 1

        for i in range(m):
            x, y, w = E[i].get( )
            E[i].x = cir[x]
            E[i].y = cir[y]
            E[i].w -= bst[y]

        time += 1
        root = cir[root]
        n = cnt

    n = org_n
    latest = [-1] * n
    for i in range(m):
        x, y, t = E[i].get_r( )
        latest[y] = max(latest[y],t)
    for i in range(m):
        x, y, t = E[i].get_r( )
        if latest[y] == t:
            head[y] = x

    return True


def get_head(W, ans=None, ex=0.0):

    n = len(W)
    W = np.array(W)
    if ans == None:
        ans = [-1] * n

    E = [ ]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            E.append(edge(j, i, W[i][j] + ex * (ans[j] != i)))

    head = [0] * n
    assert cle(E, n, len(E), n-1, head)
    return head

