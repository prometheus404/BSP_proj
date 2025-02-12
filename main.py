from entropy import entropies_for_m_range
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# compute nsr entropies
filenames = [f"nsr2/nsr{n:03}" for n in range(1, 21)]  # signals
m_range = range(2, 21)  # range for the embedding size
entropies_nsr = entropies_for_m_range(filenames, m_range)
labels = ["peEn", "RpEn", "cPE", "cRpEn", "bbEn"]

# plot entropy for various values of m
for i in range(5):
    plt.plot(m_range, entropies_nsr.mean(1)[:, i], label=labels[i])
    plt.xticks(m_range)
plt.legend(loc="best")
plt.savefig("img/entropies_for_m_range_nsr.svg")
plt.close()

# compute chf entropies
filenames = [f"chf2/chf2{n:02}" for n in range(1, 21)]
entropies_chf = entropies_for_m_range(filenames, m_range)

# plot entropy for varous values of m for chf
for i in range(len(labels)):
    plt.plot(m_range, entropies_nsr.mean(1)[:, i], label=labels[i])
    plt.xticks(m_range)
plt.legend(loc="best")
plt.savefig("img/entropies_for_m_range_chf.svg")
plt.close()


# computing the discriminating power of entropy
for i in range(len(labels)):
    p = [
        ttest_ind(entropies_nsr[m, :, i], entropies_chf[m, :, i]).pvalue
        for m in range(entropies_nsr.shape[0])
    ]
    plt.plot(m_range, p, "o", label=labels[i])
    plt.xticks(m_range)
plt.legend(loc="best")
plt.savefig("img/discriminating_power_full.svg")
plt.close()i

# computing the discriminating power of entropy for selected entropy measures
for i in range(len(labels)):
    if labels[i] in ["peEn", "bbEn"]:
        p = [
            ttest_ind(entropies_nsr[m, :, i], entropies_chf[m, :, i]).pvalue
            for m in range(entropies_nsr.shape[0])
        ]
        plt.plot(m_range, p, "o", label=labels[i])
        plt.xticks(m_range)
plt.legend(loc="best")
plt.savefig("img/discriminating_power_selected.svg")
plt.close()
