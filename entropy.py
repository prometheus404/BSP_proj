from collections import defaultdict
from math import log
from wfdb.processing import ann2rr
from tqdm import tqdm
import numpy as np
import time


##### UTILS
def _embed(timeserie, m):
    """return the embeddings of size m of the timeserie"""
    return [
        tuple([timeserie[e] for e in range(i, i + m)])
        for i in range(len(timeserie) - m + 1)
    ]


def _compute_toi(timeserie):
    """return the timeserie of indices from the input timeserie"""
    return [
        tuple([index for (index, _) in sorted(enumerate(t_i), key=lambda x: x[1])])
        for t_i in timeserie
    ]


def _probabilities(timeserie):
    """return the probabilities of each pattern in the timeserie"""
    p = defaultdict(int)
    for t in timeserie:
        p[t] += 1
    for e in p:
        p[e] /= len(timeserie)
    return p


###### PERMUTATION ENTROPY
def _peEn(J):
    """return the Shannon Entropy of the timeserie of indices"""
    p = _probabilities(J)
    return -sum([p[j_i] * log(p[j_i]) for j_i in p])


def peEn(timeserie, m):
    """return the Shannon Entropy of a timeserie using embedding size m"""
    return _peEn(_compute_toi(_embed(timeserie, m)))


def cPE(timeserie, m):
    """return the conditional permutation entropy of a timeserie using embedding size m"""
    return peEn(timeserie, m + 1) - peEn(timeserie, m)


###### RENYI PERMUTATION ENTROPY
def _rpEn(J, m=0):
    """return an unnormalized rpen with alpha = 2 if m = 0, a normalized rpen otherwise"""
    p = _probabilities(J)
    # TODO maybe since probabilities is used by both rpEn and peEn, this should be the argument
    if m == 0:
        return -log(sum([p[j_i] ** 2 for j_i in p]))
    else:
        return -(log(sum([p[j_i] ** 2 for j_i in p])) / log(m))


def rpEn(timeserie, m):
    return _rpEn(_compute_toi(_embed(timeserie, m)), m)


def cRpEn(timeserie, m):
    return (rpEn(timeserie, m + 1) - rpEn(timeserie, m)) / log(m + 1)


###### BUBBLE ENTROPY
def _bubble_sort(l):
    l = list(l)
    res = 0
    swapped = False
    for i in range(len(l)):
        for j in range(len(l) - i - 1):
            if l[j] > l[j + 1]:
                l[j], l[j + 1] = l[j + 1], l[j]
                res += 1
                swapped = True
        if not swapped:
            break
    return res, l


def _not_eff_bbEn(X):
    return _rpEn([_bubble_sort(x_i)[0] for x_i in X])


def _fast_sort(to_remove, to_add, prev_sorted):
    pos = prev_sorted.index(to_remove)
    prev_sorted = prev_sorted[:pos] + prev_sorted[pos + 1 :]
    for index, element in enumerate(prev_sorted):
        if element > to_add:
            prev_sorted.insert(index, to_add)
            return len(prev_sorted) - 1 - index - pos, prev_sorted
    return -pos, prev_sorted + [to_add]


def _bbEn(X):
    to_remove = X[0][0]
    i, prev_sorted = _bubble_sort(X[0])
    J = [i]
    sorted = [prev_sorted]
    for x_i in X[1:]:
        r, prev_sorted = _fast_sort(to_remove, x_i[-1], prev_sorted)
        sorted.append(prev_sorted)
        to_remove = x_i[0]
        J.append(r + J[-1])
    return _rpEn(J)


def bbEn(timeserie, m):
    """return bubble entropy of timeserie X using embedding dimension m"""
    return (_bbEn(_embed(timeserie, m + 1)) - _bbEn(_embed(timeserie, m))) / log(
        (m + 1) / (m - 1)
    )


def entropies_for_m_range(filenames, m_range):
    """For each entropy measure and for each signal in filenames
    compute the entropy using the range of embedding size defined.

    This function is optimized for computation of subsequent values of m: embeddings are compted
    only once and used for each entropy measure.

    TODO: for bbEn we could exploit the fact that the embeddings at dimension m are almost the same
          as the embeddings at dimension m+1 except for one element, so we could simply add to the last bbEn value for
          the embedding the ordering of the last element.
          Note however that this way we should further unpack the bbEn function.

    Args:
        filenames: The names of the files from which signals are read
        (TODO this should be an iterator on signals in order to abstract from the filetype)
        m_range: The range of values of the embedding size
    Returns:
        A three dimension (m value, signal, entropy measure) numpy array
    """

    entropies = np.zeros((len(m_range), len(filenames), 5))
    for m in tqdm(m_range):
        # res = np.zeros((len(signals), 4))
        for i, s in enumerate(filenames):
            x = ann2rr(s, "ecg")
            X = _embed(x, m)
            J = _compute_toi(X)
            entropies[m - m_range.start, i] = [
                _peEn(J),
                _rpEn(J),
                0,
                0,
                _bbEn(X),
            ]

    # compute conditional permutation entropy
    entropies[:, :, 2] = [
        entropies[m + 1, :, 0] - entropies[m, :, 0]
        if m < entropies.shape[0] - 1
        else (np.zeros(entropies.shape[1]))
        for m in range(entropies.shape[0])
    ]

    # compute conditional renyi permutation entropy
    entropies[:, :, 3] = [
        (entropies[m + 1, :, 1] - entropies[m, :, 1]) / log(m + m_range.start + 1)
        if m < entropies.shape[0] - 1
        else np.zeros(entropies.shape[1])
        for m in range(entropies.shape[0])
    ]

    # compute renyi on bubble entropy
    entropies[:, :, 4] = [
        (entropies[m + 1, :, 4] - entropies[m, :, 4])
        / log((m + m_range.start + 1) / (m + m_range.start - 1))
        if m < entropies.shape[0] - 1
        else np.zeros(entropies.shape[1])
        for m in range(entropies.shape[0])
    ]

    # normalize renyi entropy
    entropies[:, :, 1] = [
        [x / log(m + m_range.start) for x in e]
        for m, e in enumerate(entropies[:, :, 1])
    ]

    return entropies[:-1, :, :]


if __name__ == "__main__":
    x = [6, 2, 1, 4, 5, 3, 2, 1, 4, 3, 2, 1]  # timeserie
    print(x)
    X = _embed(x, 3)
    print(X)
    print(_not_eff_bbEn(X))
    print(_bbEn(X))

    # print(bubble_sort(x[:]))
    # print(x)
    # print(bubble_sort([1]))
    x = ann2rr("nsr2/nsr001", "ecg")
    for m in range(2, 20):
        x_m = _embed(x, m)
        start_time = time.time()
        e2 = _bbEn(x_m)
        t2 = time.time() - start_time

        start_time = time.time()
        e1 = _not_eff_bbEn(x_m)
        t1 = time.time() - start_time

        print(f"-------------m={m}----------")
        print("efficient:", "res", e2, "time", t2)
        print("not efficient:", "res", e1, "time", t1)
        assert e1 == e2

    # x2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # x2 = ann2rr("chf2/chf201", "ecg")
    # x3 = list(reversed(x2))
    # # print(x3)
    # m = 7  # embed ded space size
    # X = embed(x, m)
    # J = compute_toi(X)
    # # print("timeserie of indices: ", J)
    # print("entropy:", peEn(J))
    # print("entropy of second signal: ", peEn(compute_toi(embed(x2, m))))
    # print("entropy of reversed second signal: ", peEn(compute_toi(embed(x3, m))))
    # for i in tqdm(range(1, 3)):
    #     # name = f"chf2/chf2{i:02}"
    #     name = f"nsr2/nsr{i:03}"
    #     x = ann2rr(name, "ecg")
    #     plt.plot([peEn(compute_toi(embed(x, m))) for m in tqdm(range(1, 20))])
    # # plt.plot([peEn(compute_toi(embed(x2, m))) for m in range(1, 20)])
    # plt.show()
