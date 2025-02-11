from entropy import embed, compute_toi, peEn, rpEn, bbEn
from wfdb.processing import ann2rr
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from scipy.stats import ttest_ind


def entropy_for_m_range(filenames, m_range):
    entropies = np.zeros((len(m_range), len(filenames), 5))
    for m in tqdm(m_range):
        # res = np.zeros((len(signals), 4))
        for i, s in enumerate(filenames):
            x = ann2rr(s, "ecg")
            X = embed(x, m)
            J = compute_toi(X)
            entropies[m - m_range.start, i] = [
                peEn(J),
                rpEn(J),
                0,
                0,
                bbEn(X),
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
        (entropies[m + 1, :, 1] - entropies[m, :, 1]) / math.log(m + m_range.start + 1)
        if m < entropies.shape[0] - 1
        else np.zeros(entropies.shape[1])
        for m in range(entropies.shape[0])
    ]

    # compute renyi on bubble entropy
    entropies[:, :, 4] = [
        (entropies[m + 1, :, 4] - entropies[m, :, 4])
        / math.log((m + m_range.start + 1) / (m + m_range.start - 1))
        if m < entropies.shape[0] - 1
        else np.zeros(entropies.shape[1])
        for m in range(entropies.shape[0])
    ]
    return entropies

    # normalize renyi entropy
    entropies[:, :, 1] = [
        [x / math.log(m + m_range.start) for x in e]
        for m, e in enumerate(entropies[:, :, 1])
    ]


# compute nsr entropies
filenames = [f"nsr2/nsr{n:03}" for n in range(1, 21)]  # signals
m_range = range(2, 21)  # range for the embedding size
entropies_nsr = entropy_for_m_range(filenames, m_range)

labels = ["peEn", "RpEn", "cPE", "cRpEn", "bben"]
# # plot entropy for various values of m
# for i in range(5):
#     plt.plot(m_range, entropies_nsr.mean(1)[:, i], label=labels[i])
#     plt.xticks(m_range)
# plt.legend(loc="best")
# plt.show()
#
# compute chf entropies
filenames = [f"chf2/chf2{n:02}" for n in range(1, 21)]
entropies_chf = entropy_for_m_range(filenames, m_range)
# # plot entropy for varous values of m for chf
# for i in range(len(labels)):
#     plt.plot(m_range, entropies_nsr.mean(1)[:, i], label=labels[i])
#     plt.xticks(m_range)
# plt.legend(loc="best")
# plt.show()


# computing the discriminating power of entropy
for i in range(len(labels)):
    p = [
        ttest_ind(entropies_nsr[m, :, i], entropies_chf[m, :, i]).pvalue
        for m in range(entropies_nsr.shape[0])
    ]
    plt.plot(m_range, p, "o", label=labels[i])
    plt.xticks(m_range)
plt.legend(loc="best")
plt.show()
