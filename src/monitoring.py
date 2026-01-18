import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def population_stability_index(expected, actual, bins=10):
    expected_perc, _ = np.histogram(expected, bins=bins)
    actual_perc, _ = np.histogram(actual, bins=bins)

    expected_perc = expected_perc / len(expected)
    actual_perc = actual_perc / len(actual)

    psi = np.sum(
        (actual_perc - expected_perc) *
        np.log((actual_perc + 1e-6) / (expected_perc + 1e-6))
    )
    return psi


def detect_drift(train_df, new_df, features):
    drift_report = {}

    for col in features:
        psi = population_stability_index(
            train_df[col].dropna(),
            new_df[col].dropna()
        )
        drift_report[col] = psi

    return drift_report
