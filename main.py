from entropy import peEn, rpEn, cPE, cRpEn, bbEn, sampEn, entropies_for_m_range
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from wfdb.processing import ann2rr
from utils import timed, SignalIterator


################## ENTROPIES AND SIGNAL LENGTH #################
# SaEn -> m=2, r=0.2 # peEn, RpEn -> m=9 # cPE, cRpEn-> m=5 # bbEn -> m=10
functions = [
    ("peEn", lambda t: peEn(t, 10)),
    ("rpEn", lambda t: rpEn(t, 10)),
    ("cPE", lambda t: cPE(t, 5)),
    ("cRpEn", lambda t: cRpEn(t, 5)),
    ("bbEn", lambda t: bbEn(t, 10)),
    # ("sampEn", lambda t: sampEn(t, 2, 0.2)),
]
timeserie = ann2rr("nsr2/nsr001", "ecg")  # TODO average results from various signals
n_range = [50 * (2**exp) for exp in range(12)]
t_fig = plt.figure()
e_fig = plt.figure()
t_ax = t_fig.add_subplot(111)
e_ax = e_fig.add_subplot(111)

for name, f in functions:
    times = []
    entropies = []
    for n in n_range:
        if name == "sampEn" and n > 50 * (2**8):
            break
        t, e = timed(f, timeserie[:n])
        times.append(t)
        entropies.append(e)
    if name == "sampEn":
        t_ax.plot(n_range[:9], times, label=name)
        e_ax.plot(n_range[:9], entropies, label=name)
    else:
        t_ax.plot(n_range, times, label=name)
        e_ax.plot(n_range, entropies, label=name)

e_ax.set_title("Entropy for various signal length")
t_ax.set_title("Execution time for various signal length")
t_ax.set_xlabel("Signal length")
e_ax.set_xlabel("Signal length")
t_ax.set_ylabel("Execution time (s)")
e_ax.set_ylabel("Entropy")
t_ax.legend(loc="best")
e_ax.legend(loc="best")
t_fig.savefig("img/times_for_n_range.svg")
e_fig.savefig("img/entropies_for_n_range.svg")
plt.close("all")
# zoomed
timeserie = ann2rr("nsr2/nsr001", "ecg")  # TODO average results from various signals
n_range = range(50, 10000, 100)
e_fig = plt.figure()
e_ax = e_fig.add_subplot(111)

for name, f in functions:
    entropies = []
    for n in n_range:
        _, e = timed(f, timeserie[:n])
        entropies.append(e)
    e_ax.plot(n_range, entropies, label=name)

e_ax.set_title("Entropy for various signal length (zoomed)")
e_ax.set_xlabel("Signal length")
e_ax.set_ylabel("Entropy")
e_ax.legend(loc="upper right")
e_fig.savefig("img/entropies_for_n_range_zoom.svg")
plt.close("all")
quit()
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
