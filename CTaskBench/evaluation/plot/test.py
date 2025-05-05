import collections
import json
import os
import time
from bisect import bisect_right

import numpy as np
from scipy.stats import skewnorm, truncnorm
from scipy.integrate import quad


def monte_carlo_integral(func, observation, scale, num_samples=10):
    upper_bound = skewnorm.ppf(0.975, alpha, loc=loc, scale=scale)
    samples = np.random.uniform(observation, upper_bound, num_samples)

    function_values = func(samples)
    mean_of_function_values = np.mean(function_values)
    integral_approximation = (np.max(samples) - observation) * mean_of_function_values

    return integral_approximation


def truncated_skew_normal_mean(alpha: float, loc: float, scale: float, observation: float, monte_carlo: bool = True):
    """
    Calculate the mean of a skew-normal distribution truncated above the given observation.

    :param alpha: Skewness parameter of the skew-normal distribution.
    :param loc: Location parameter (mean of the distribution).
    :param scale: Scale parameter (standard deviation of the distribution).
    :param observation: The observed value, which acts as the truncation point.
    :return: The mean of the truncated skew-normal distribution.

    """
    # Calculate the cumulative distribution function (CDF) at the observation
    t = time.time()
    cdf_value = skewnorm.cdf(observation, alpha, loc, scale)
    t1 = time.time() - t

    # Calculate the probability density function (PDF) at the observation
    # pdf_value = skewnorm.pdf(observation, alpha, loc, scale)

    # Define the integrand function for computing the truncated mean
    def integrand(x):
        return x * skewnorm.pdf(x, alpha, loc, scale)

    # Calculate the integral of the skew-normal distribution from the observation to positive infinity
    if not monte_carlo:
        integral, _ = quad(integrand, observation, np.inf, epsabs=0.1, epsrel=0.1)
    else:
        t = time.time()
        integral = monte_carlo_integral(integrand, observation, scale)
        t2 = time.time() - t
        print(f"cdf {t1 / (t1 + t2) * 100:.2f}% monte_carlo {t2 / (t1 + t2) * 100:.2f}%")

    # Calculate the mean of the truncated distribution using the integral and the CDF value
    truncated_mean = integral / (1 - cdf_value)

    return truncated_mean

    # if __name__ == '__main__':
    #
    #     # 示例
    #     alpha = -2
    #     loc = 5
    #     scale = 1
    #     observation = 2
    #
    #     trunc_norm_dist = truncnorm((observation - loc) / scale,
    #                                 np.inf,
    #                                 loc=loc,
    #                                 scale=scale)
    #     print(f"observation: {observation}",
    #           trunc_norm_dist.mean(),
    #           truncated_skew_normal_mean(alpha, loc, scale, observation),
    #           truncated_skew_normal_mean(alpha, loc, scale, observation, False),
    #           )
    #
    #     timer = time.time()
    #     for _ in range(50):
    #         truncated_skew_normal_mean(alpha, loc, scale, observation)
    #     print(f"Time: {(time.time() - timer) * 1000:.2f}ms")
    #
    #     timer = time.time()
    #     for _ in range(50):
    #         truncated_skew_normal_mean(alpha, loc, scale, observation, False)
    #     print(f"Time: {(time.time() - timer) * 1000:.2f}ms")


def generate_skew_normal_samples(alpha: float, loc: float, scale: float, num_samples: int = 100):
    return skewnorm.rvs(alpha, loc, scale, size=num_samples).tolist()


def get_directory_info(path='.'):
    total_files = 0
    total_dirs = 0
    total_size = 0

    for root, dirs, files in os.walk(path):
        print(root, dirs, files)
        # Increment directory count
        total_dirs += len(dirs)
        # Increment file count
        total_files += len(files)
        # Calculate the total size of the files
        total_size += sum(os.path.getsize(os.path.join(root, file)) for file in files)

    return total_dirs, total_files, total_size

    # path = '/mnt/jfs/fashion-dataset/images'  # Current directory
    # dirs, files, size = get_directory_info(path)
    #
    # print(f"Total directories: {dirs}")
    # print(f"Total files: {files}")
    # print(f"Total size: {size / (1024 ** 2):.2f} MB {size / (1024 ** 3):.2f} GB")  # Convert bytes to MB for display


if __name__ == '__main__':
    # with open("../results/sched_sjf_window20_task500_try0_intensity1.5/Hermes-Gittins.json", "r") as f:
    with open("../results/sched_sjf_window20_task500_try0_intensity1.5/Hermes-Gittins.json", "r") as f:
        res = json.load(f)
    # print(res)
    jct_rec_slo = {
        task_id: [data] + res["task_SLO"][task_id]
        for task_id, data in res["task_completion_time"].items()
        if "multiturn_conversations" not in task_id
    }
    # print(json.dumps(jct_rec_slo, indent=4))

    stat = {}
    for task_id, data in jct_rec_slo.items():
        # if data[2] is None:
        #     continue
        app = task_id.split("--")[0]
        if app not in stat:
            stat[app] = []
        stat[app].append(data)
    # print(json.dumps(stat, indent=4))

    avg_jct = {
        app: [
            np.mean([data[0] for data in stats]),
            np.mean([data[1] for data in stats]),
        ]
        for app, stats in stat.items()
    }
    print(json.dumps(avg_jct, indent=4))

# {
#     "factool_math": 8.419965690182101,
#     "react_fever": 9.284382059460594,
#     "factool_kbqa": 46.26995518130641,
#     "react_alfw": 28.267542907169886,
#     "factool_code": 71.33857888760774,
#     "hugginggpt": 51.66593565940857,
#     "code_feedback": 467.3393848159096,
#     "langchain_mapreduce": 627.6235981548534,
#     "got_docmerge": 2145.5747520128884
# }

# {
#     "factool_math": 7.022946949928038,
#     "react_fever": 6.228117034548805,
#     "factool_kbqa": 17.261207526729955,
#     "react_alfw": 26.29913571902684,
#     "factool_code": 16.202185164327208,
#     "hugginggpt": 47.71458277702332,
#     "code_feedback": 93.63589624925093,
#     "langchain_mapreduce": 204.32263580490562,
#     "got_docmerge": 2472.4943288962045
# }

