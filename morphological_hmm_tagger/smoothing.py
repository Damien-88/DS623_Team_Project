"""Probability smoothing helpers for the HMM tagger."""

from collections import Counter


def add_alpha_probability(count, total, support_size, alpha=0.1):
    """Return Laplace/add-alpha smoothed probability."""
    if support_size <= 0:
        return 0.0
    return (count + alpha) / (total + alpha * support_size)


def frequency_of_frequencies(counter):
    """Build a frequency-of-frequency table from a count mapping."""
    return Counter(counter.values())


def good_turing_adjusted_count(count, fof, cutoff=5):
    """Return an approximate Good-Turing adjusted count."""
    if count < 0:
        raise ValueError("Counts must be non-negative.")

    if count == 0:
        return float(fof.get(1, 0))

    if count <= cutoff:
        current_frequency = fof.get(count, 0)
        next_frequency = fof.get(count + 1, 0)
        if current_frequency > 0 and next_frequency > 0:
            return (count + 1) * next_frequency / current_frequency

    return float(count)


def good_turing_probability(count, counter, support_size, cutoff=5):
    """Approximate the Good-Turing probability for a single event."""
    total = sum(counter.values())
    if total <= 0 or support_size <= 0:
        return 0.0

    fof = frequency_of_frequencies(counter)
    adjusted_total = sum(good_turing_adjusted_count(value, fof, cutoff=cutoff) for value in counter.values())
    adjusted_total = adjusted_total if adjusted_total > 0 else float(total)

    if count == 0:
        unseen_mass = fof.get(1, 0) / total
        unseen_types = max(support_size - len(counter), 1)
        return unseen_mass / unseen_types

    adjusted_count = good_turing_adjusted_count(count, fof, cutoff=cutoff)
    return adjusted_count / adjusted_total