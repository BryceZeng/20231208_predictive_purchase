from scipy import stats
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from scipy.signal import correlate


def train_explainer(df):
    model = None
    return model


def calculate_slope(row):
    data = [
        row["CCH_lag_2"],
        row["CCH_lag_1"],
        row["CCH"],
        row["pred1"],
        row["pred2"],
        row["pred3"],
        row["pred4"],
        row["pred5"],
        row["pred6"],
    ]
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        range(len(data)), data
    )
    period_to_cross_zero = abs(intercept / slope) if slope < 0 else None
    return slope, p_value, period_to_cross_zero


def find_transform(series):
    # Drop NA values and get the length once
    series = series.dropna()
    series = np.trim_zeros(series, "f")
    series_len = len(series)

    # Check if series length is less than 18, if so return None values
    if series_len < 18:
        return None, None, None, None, None, None, None

    # Calculate acf_values only once for each part of the series
    acf_values = acf(series[-12:], nlags=6)
    acf_values2 = acf(series[-18:-6], nlags=6)

    # Calculate best_period only once for each part of the series
    best_period = np.argmax(acf_values[1:]) + 1
    best_period2 = np.argmax(acf_values2[1:]) + 1

    # Decompose the series only once for each part of the series
    result = seasonal_decompose(series[-12:], model="additive", period=best_period)
    result2 = seasonal_decompose(series[-18:-6], model="additive", period=best_period2)

    # Calculate diff_phase and amplitude_max only once
    diff_phase = best_period - best_period2
    # amplitude_max = min(result.seasonal.max(), result2.seasonal.max())
    amplitude_max = result2.seasonal.max()
    seasonal_shift = np.argmax(
        correlate(result.seasonal.tolist(), result2.seasonal.tolist())
    )

    return (
        diff_phase,
        amplitude_max,
        best_period,
        best_period2,
        result.seasonal.tolist(),
        result2.seasonal.tolist(),
        seasonal_shift,
    )


def predict_explainer(df, model):
    return df
