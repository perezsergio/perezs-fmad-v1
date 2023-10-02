import numpy as np
import scipy.stats as stats


def get_confidence_interval(sample, confidence_level=0.95, distribution="StudentsT"):
    """
    Given a one-variable size-m sample and the desired confidence level, returns the confidence interval of the mean.

    Args:
        sample : (numpy.ndarray Shape (m)) Sample
        confidence_level: (float in [0,1]) Probability that the mean of the population falls inside the confidence interval
        distribution: ("StudentsT" or "Normal") Use "StudentsT" for small samples, and "Normal" for big samples


    Returns:
        confidence_interval: (numpy.ndarray Shape (2)) The interval where the mean of the population falls, with a probability of p = confidence_level
        delta: (float): half-width of the interval

    """

    # Probability that the population mean falls outside the confidence interval
    alpha = 1 - confidence_level

    # Use the specified distribution to calculate the critical point
    if distribution == "StudentsT":
        degrees_of_freedom = sample.size - 1
        critical_point = stats.t.ppf(1 - alpha / 2, degrees_of_freedom)
    elif distribution == "Normal":
        critical_point = stats.norm.ppf(1 - alpha / 2, loc=0, scale=1)
    else:
        raise ValueError("distribution must be either 'StudentsT' or 'Normal")

    # Calculate the half-width (delta) and the confidence interval
    delta = critical_point * sample.std() / np.sqrt([sample.size])
    confidence_interval = sample.mean() + np.array([-1, +1]) * delta

    return confidence_interval, delta
