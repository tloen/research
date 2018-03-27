import numpy as np

def normalize(x):
    x /= np.sum(x)

def sinkhorn_operator(X, T):
    S = np.exp(X)
    n, m = S.shape
    for t in range(T):
        # T_r
        for i in range(n):
            normalize(S[i, :])
        # T_c
        for j in range(m):
            normalize(S[:, j])
    return S

def gumbels(n):
    U = np.random.rand(n)
    return -np.log(-np.log(U))

def concrete_distribution(alpha, tau):
    n = len(alpha)
    G = gumbels(n)
    vect = np.exp((np.log(alpha) + G) / tau)
    vect /= np.sum(vect)
    return vect

def frobenius_product(A, B):
    return np.trace(A.T.dot(B))

def gumbel_sample(alpha):
    n = len(alpha)
    G = gumbels(n)
    vect = np.log(alpha) + G
    return vect

def sinkhorn_approx(alpha, n = 1000):
    res = np.zeros((len(alpha), len(alpha)))
    for _ in range(n):
        order = gumbel_sample(alpha)
        reorder = np.flip(np.argsort(order), axis=0)
        for j in range(len(alpha)):
            res[j, reorder[j]] += 1
    res /= n
    return res

alpha = [4, 3, 2, 1]

print(sinkhorn_approx(alpha, 100))



'''
X = np.random.rand(4, 4)
print("orig: ", X)

for ltau in range(1, 5):
    tau = np.exp(-ltau)
    print(sinkhorn_operator(X/tau, 50))
'''

