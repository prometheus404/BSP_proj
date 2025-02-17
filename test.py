from os import cpu_count
from entropy import peEn, rpEn, cPE, cRpEn, bbEn, sampEn, entropies_for_m_range
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from utils import timed, SignalIterator
import numpy as np

# from sampen import sampen2
################## ENTROPIES AND SIGNAL LENGTH #################
# SaEn -> m=2, r=0.2 # peEn, RpEn -> m=9 # cPE, cRpEn-> m=5 # bbEn -> m=10
functions = [
    ("peEn", lambda t: peEn(t, 10)),
    ("rpEn", lambda t: rpEn(t, 10)),
    ("cPE", lambda t: cPE(t, 5)),
    ("cRpEn", lambda t: cRpEn(t, 5)),
    ("bbEn", lambda t: bbEn(t, 10)),
    # ("sampEn", lambda t: sampen2(t, 2, 0.2)[2][1]),
    # ("sampEn", lambda t: sampEn(t, 2, 0.2)),
]

n_signals = 20
n_sample_range = [50 * (2**exp) for exp in range(12)]
it = SignalIterator(n_signals, "nsr2/nsr{:03d}", list(range(1, 54)))
entropies = np.zeros((n_signals, len(functions), len(n_sample_range)))
times = np.zeros((n_signals, len(functions), len(n_sample_range)))

for tn, timeserie in enumerate(it):
    print(f"signal {tn}/{n_signals}")
    for fn, (name, f) in enumerate(functions):
        print(name)
        for nn, n in enumerate(n_sample_range):
            times[tn, fn, nn], entropies[tn, fn, nn] = timed(f, timeserie[:n])

for i, (name, _) in enumerate(functions):
    plt.plot(n_sample_range, entropies[:, i, :].mean(0), label=name)
plt.title("Entropy for various signal length")
plt.xlabel("Signal length")
plt.ylabel("Entropy")
plt.legend(loc="best")
plt.savefig("img/entropies_for_n_range.svg")
plt.close()
for i, (name, _) in enumerate(functions):
    plt.plot(n_sample_range, times[:, i, :].mean(0), label=name)
plt.title("Execution time for various signal length")
plt.xlabel("Signal length")
plt.ylabel("Execution time (s)")
plt.legend(loc="best")
plt.savefig("img/times_for_n_range.svg")
plt.close()

# zoomed
n_signals = 20
n_sample_range = range(50, 10000, 100)
it = SignalIterator(n_signals, "nsr2/nsr{:03d}", list(range(1, 54)))
entropies = np.zeros((n_signals, len(functions), len(n_sample_range)))

for tn, timeserie in enumerate(it):
    for fn, (name, f) in enumerate(functions):
        for nn, n in enumerate(n_sample_range):
            entropies[tn, fn, nn] = f(timeserie[:n])

for i, (name, _) in enumerate(functions):
    plt.plot(n_sample_range, entropies[:, i, :].mean(0), label=name)

plt.title("Entropy for various signal length (zoomed)")
plt.xlabel("Signal length")
plt.ylabel("Entropy")
plt.legend(loc="upper right")
plt.savefig("img/entropies_for_n_range_zoom.svg")
plt.close()

################## NSR ENTROPIES #################
n_signals = 20
# filenames = [f"nsr2/nsr{n:03}" for n in range(1, 21)]  # signals
it = SignalIterator(n_signals, "nsr2/nsr{:03d}", list(range(1, 54)))
m_range = range(2, 21)  # range for the embedding size
entropies_nsr = entropies_for_m_range(it, m_range)
labels = ["peEn", "RpEn", "cPE", "cRpEn", "bbEn"]
# plot entropy for various values of m
for i in range(5):
    plt.plot(m_range, entropies_nsr.mean(1)[:, i], label=labels[i])
    plt.xticks(m_range)
plt.title("Entropy for various embedding dimension m (nsr)")
plt.xlabel("Value of m")
plt.ylabel("Entropy")
plt.legend(loc="best")
plt.savefig("img/entropies_for_m_range_nsr.svg")
plt.close()

# timed test
n_signals = 20

functions = [peEn, rpEn, cPE, cRpEn, bbEn]
labels = ["peEn", "RpEn", "cPE", "cRpEn", "bbEn"]
times = np.zeros((n_signals, len(functions), len(m_range)))
for fn, f in enumerate(functions):
    print(f)
    for mn, m in enumerate(m_range):
        print(m)
        for tn, timeserie in enumerate(it):
            times[tn, fn, mn], _ = timed(f, timeserie, m)
            print(times[tn, fn, mn])

for i, name in enumerate(labels):
    plt.plot(m_range, times[:, i, :].mean(0), label=name)
plt.title("Execution time for various embedding dimension m")
plt.xlabel("Value of m")
plt.ylabel("Execution time (s)")
plt.legend(loc="center right")
plt.savefig("img/times_for_m_range.svg")
plt.close()


################## CHF ENTROPIES #################
# filenames = [f"chf2/chf2{n:02}" for n in range(1, 21)]
it = SignalIterator(n_signals, "chf2/chf2{:02d}", list(range(1, 29)))
entropies_chf = entropies_for_m_range(it, m_range)

# plot entropy for varous values of m for chf
for i in range(len(labels)):
    plt.plot(m_range, entropies_nsr.mean(1)[:, i], label=labels[i])
    plt.xticks(m_range)

plt.title("Entropy for various embedding dimension m (chf)")
plt.xlabel("Value of m")
plt.ylabel("Entropy")
plt.legend(loc="best")
plt.savefig("img/entropies_for_m_range_chf.svg")
plt.close()


################## DISCRIMINATING POWER #################
for i in range(len(labels)):
    p = [
        ttest_ind(entropies_nsr[m, :, i], entropies_chf[m, :, i]).pvalue
        for m in range(entropies_nsr.shape[0])
    ]
    plt.plot(m_range, p, "o", label=labels[i])
    plt.xticks(m_range)

plt.title("Discriminating power for various values of m")
plt.xlabel("Embedding dimension (m)")
plt.ylabel("p-value")
plt.legend(loc="best")
plt.savefig("img/discriminating_power_full.svg")
plt.close()

# computing the discriminating power of entropy for selected entropy measures
for i in range(len(labels)):
    if labels[i] in ["peEn", "bbEn"]:
        p = [
            ttest_ind(entropies_nsr[m, :, i], entropies_chf[m, :, i]).pvalue
            for m in range(entropies_nsr.shape[0])
        ]
        plt.plot(m_range, p, "o", label=labels[i])
        plt.xticks(m_range)

plt.title("Discriminating power for various values of m (detail on peEn and bbEn)")
plt.xlabel("Embedding dimension (m)")
plt.ylabel("p-value")
plt.legend(loc="best")
plt.savefig("img/discriminating_power_selected.svg")
plt.close()
