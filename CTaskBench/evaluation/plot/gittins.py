from bisect import bisect_right


class Dist:
    bin_edges: list = []
    gittins: dict = {}


def update_cache(dist, samples):
    # counts is the number of samples in each bin
    # bin_edges is the left edges of the bins
    # bin_sum is the sum of the values in each bin
    counts, dist.bin_edges, bin_sum = generate_histogram(samples)
    val_suffix_sum = calculate_suffix_sum(bin_sum)
    cnt_suffix_sum = calculate_suffix_sum(counts)

    for i, v in enumerate(dist.bin_edges[:-1]):
        gittins_indexes = []
        for j in range(i + 1, len(val_suffix_sum)):
            val_interval_sum = val_suffix_sum[i] - val_suffix_sum[j]
            cnt_interval_sum = cnt_suffix_sum[i] - cnt_suffix_sum[j]
            E = val_interval_sum / cnt_interval_sum
            P = cnt_interval_sum / cnt_suffix_sum[i]
            gittins_indexes.append(E / P)
        dist.gittins[v] = min(gittins_indexes)


def get_gittins_index(dist, a):
    idx = binary_search(dist.bin_edges, a)
    if idx == len(dist.bin_edges):
        return a
    return dist.gittins[dist.bin_edges[idx]]
