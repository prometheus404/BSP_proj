from collections import defaultdict
from math import log

x = [6, 2, 1, 4, 5, 3, 2, 1, 4, 3, 2, 1]  # timeserie
m = 3  # embedded space size
N = len(x)
X = [[x[e] for e in range(i, i + m)] for i in range(N - m + 1)]
print("timeserie: ", X)
J = [[index for (index, _) in sorted(enumerate(X_i), key=lambda x: x[1])] for X_i in X]
print("timeserie of indices: ", J)


def probabilities(timeserie):
    """computes the probabilities of each pattern in the timeserie"""
    p = defaultdict(int)
    for t in timeserie:
        p[tuple(t)] += 1
    for e in p:
        p[tuple(e)] /= len(timeserie)
    return p


def peEn(J):
    """computes the Shannon Entropy of the timeserie of indices"""
    p = probabilities(J)
    return -sum([p[j_i] * log(p[j_i]) for j_i in map(tuple, J)])


# print(probabilities(J))
print(peEn(J))
