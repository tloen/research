import numpy as np

def oddeven_merge(lo, hi, r):
    step = r * 2
    if step < hi - lo:
        yield from oddeven_merge(lo, hi, step)
        yield from oddeven_merge(lo + r, hi, step)
        yield from [(i, i + r) for i in range(lo + r, hi - r, step)]
    else:
        yield (lo, lo + r)

def oddeven_merge_sort_range(lo, hi):
    """ sort the part of x with indices between lo and hi.

    Note: endpoints (lo and hi) are included.
    """
    if (hi - lo) >= 1:
        # if there is more than one element, split the input
        # down the middle and first sort the first and second
        # half, followed by merging them.
        mid = lo + ((hi - lo) // 2)
        yield from oddeven_merge_sort_range(lo, mid)
        yield from oddeven_merge_sort_range(mid + 1, hi)
        yield from oddeven_merge(lo, hi, 1)

def oddeven_merge_sort(length):
    """ "length" is the length of the list to be sorted.
    Returns a list of pairs of indices starting with 0 """
    yield from oddeven_merge_sort_range(0, length - 1)

def compare_and_swap(x, a, b):
    if x[a] > x[b]:
        x[a], x[b] = x[b], x[a]
    print(x[a], x[b])

def sinkhorn_swap(x, a, b):
    x[a], x[b] = (x[a] ** 2 + x[b] ** 2) / (x[a] + x[b]), 2 * x[a] * x[b] / (x[a] + x[b])

'''
def continuous_max(a, b, t):
    return (1/t) * np.log(np.sum(np.exp(a * t) + np.exp(b * t)))

def soft_compare_and_swap(x, a, b, t):
    high = continuous_max(x[a], x[b], t)
    x[a], x[b] = x[a] + x[b] - high, high
    print(x[a], x[b])
'''

for _ in range(100):
    data = [np.exp(i) for i in range(8)]
    data = np.random.permutation(data).tolist()
    pairs_to_compare = list(oddeven_merge_sort(len(data)))
    for i in pairs_to_compare: sinkhorn_swap(data, *i)
    print([np.log(i) for i in data])