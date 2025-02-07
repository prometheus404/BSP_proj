from entropy import embed, compute_toi, peEn, rpEn, rpEnN
from wfdb.processing import ann2rr
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math


signals = [f"nsr2/nsr{n:03}" for n in range(1, 21)]  # signals
m_range = (2, 20)  # range for the embedding size

entropies = np.zeros((m_range[1] - m_range[0], len(signals), 4))
for m in tqdm(range(*m_range)):
    res = np.zeros((len(signals), 4))
    for i, s in enumerate(signals):
        x = ann2rr(s, "ecg")
        X = embed(x, m)
        J = compute_toi(X)
        entropies[m - m_range[0], i] = [
            peEn(J),
            rpEn(J),
            0,  # rpEnN(J, m),
            0,
        ]
        entropies[m - m_range[0]]
# print(entropies)

# compute conditional permutation entropy
entropies[:, :, 2] = [
    entropies[m + 1, :, 0] - entropies[m, :, 0]
    if m < entropies.shape[0] - 1
    else (np.zeros(entropies.shape[1]))
    for m in range(entropies.shape[0])
]

# compute conditional renyi permutation entropy
x = [
    (entropies[m + 1, :, 1] - entropies[m, :, 1]) / math.log(m + m_range[0] + 1)
    if m < entropies.shape[0] - 1
    else np.zeros(entropies.shape[1])
    for m in range(entropies.shape[0])
]
print(x)
entropies[:, :, 3] = x
# compute renyi on bubble entropy
# normalize renyi entropy
# entropies[:, :, 1] = [math.log(e) for e in entropies[:, :, 1]]
# entropies[:, :, 3] = np.log(entropies[:, :, 1])
entropies[:, :, 1] = [
    [x / math.log(m + m_range[0]) for x in e] for m, e in enumerate(entropies[:, :, 1])
]
# print(entropies)
for i in range(4):
    plt.plot(range(*m_range), entropies.mean(1)[:, i])
    plt.xticks(range(*m_range))
plt.show()
