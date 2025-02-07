from collections import defaultdict
from math import log
from wfdb.processing import ann2rr
import matplotlib.pyplot as plt
from tqdm import tqdm


def embed(timeserie, m):
    """returns a the embeddings of size m of the timeserie"""
    return [
        tuple([timeserie[e] for e in range(i, i + m)])
        for i in range(len(timeserie) - m + 1)
    ]


def compute_toi(timeserie):
    """computes the timeserie of indices from the input timeserie"""
    return [
        tuple([index for (index, _) in sorted(enumerate(t_i), key=lambda x: x[1])])
        for t_i in timeserie
    ]


def probabilities(timeserie):
    """computes the probabilities of each pattern in the timeserie"""
    p = defaultdict(int)
    for t in timeserie:
        p[t] += 1
    for e in p:
        p[e] /= len(timeserie)
    # print(p)
    return p


def peEn(J):
    """computes the Shannon Entropy of the timeserie of indices"""
    p = probabilities(J)
    return -sum([p[j_i] * log(p[j_i]) for j_i in p])


# print(probabilities(J))
if __name__ == "__main__":
    x = [6, 2, 1, 4, 5, 3, 2, 1, 4, 3, 2, 1]  # timeserie
    x = ann2rr("nsr2/nsr001", "ecg")
    # plt.plot(x)
    # plt.yscale("log")
    # plt.show()
    x2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    x2 = ann2rr("chf2/chf201", "ecg")
    x3 = list(reversed(x2))
    # print(x3)
    m = 7  # embedded space size
    X = embed(x, m)
    J = compute_toi(X)
    # print("timeserie of indices: ", J)
    print("entropy:", peEn(J))
    print("entropy of second signal: ", peEn(compute_toi(embed(x2, m))))
    print("entropy of reversed second signal: ", peEn(compute_toi(embed(x3, m))))
    for i in tqdm(range(1, 29)):
        # name = f"chf2/chf2{i:02}"
        name = f"nsr2/nsr{i:03}"
        x = ann2rr(name, "ecg")
        plt.plot([peEn(compute_toi(embed(x, m))) for m in tqdm(range(1, 20))])
    # plt.plot([peEn(compute_toi(embed(x2, m))) for m in range(1, 20)])
    plt.show()
