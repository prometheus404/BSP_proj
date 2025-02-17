from wfdb.processing import ann2rr
from time import time
import random

random.seed(42)


def timed(func, *args):
    start = time()
    res = func(*args)
    return time() - start, res


class SignalIterator:
    """return one at a time n signals randomly extracted from a given rule to construct filenames"""

    def __init__(self, n, filename_expr, arg_value_list):
        self.n = n
        self.filename_expr = filename_expr
        self.arg_value_list = arg_value_list
        self.chosen = random.sample(self.arg_value_list, self.n)
        print(self.chosen)

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.counter < self.n:
            self.counter += 1
            return ann2rr(
                self.filename_expr.format(self.chosen[self.counter - 1]), "ecg"
            )
        else:
            raise StopIteration

    def __len__(self):
        return self.n


if __name__ == "__main__":
    it = SignalIterator(10, "nsr2/nsr{:03d}", list(range(1, 54)))
    for i, j in enumerate(it):
        print(i, j)

    it = SignalIterator(10, "nsr2/nsr{:03d}", list(range(1, 54)))
    from entropy import entropies_for_m_range

    entropies_for_m_range(it, range(1, 3))
