from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import modeling_class
# import modeling_explain
import helper
import modeling_ts
import dill
import numpy as np
import pandas as pd

current_date = datetime.today() - relativedelta(days=10)
min_date = current_date - relativedelta(months=24)
current_date = current_date.strftime("%Y-%m") + "-01"
min_date = min_date.strftime("%Y-%m") + "-01"
current_date = "2023-11-01"
# For 1st stage - ts prediction
with open("predictor11.pkl", "rb") as f:
    model_ts = dill.load(f)

# For 2nd stage - ts across period
with open("model_list2.pkl", "rb") as f:
    model_class = dill.load(f)


dtypes = {"CUSTOMER_NUMBER": str, "SHIP_TO": str}
df = pd.read_csv("data/data_6.csv")
df["POSTING_DATE"] = pd.to_datetime(df["POSTING_DATE"])
# df.head()
df["POSTING_DATE"].max()
df.columns

df = helper.clean_data(df)
df = helper.create_lags(df)
df_time = modeling_ts.predict_timeseries(df, model_ts, start_date=current_date)
# df_time = df_time2.copy()
df_time.reset_index(inplace=True)

df_time[df_time["CUSTOMER_SHIPTO"] == "1628667_1628667"]

# df_time = pd.read_csv("temp.csv")
df_p = modeling_class.predict_classifer(
    df, df_time, model_class, start_date=current_date
)


from scipy import stats


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
    if slope < 0:
        period_to_cross_zero = abs(intercept / slope)
    else:
        period_to_cross_zero = None
    max_cch = max([row["CCH_lag_2"], row["CCH_lag_1"], row["CCH"]])
    percent_decline = slope / max_cch

    return slope, percent_decline, max_cch, p_value, period_to_cross_zero


df_p[["slope", "percent_decline", "max_cch", "p_value", "cross_zero"]] = df_p.apply(
    calculate_slope, axis=1, result_type="expand"
)

df_out = pd.melt(
    df_p,
    id_vars=["CUSTOMER_SHIPTO", "POSTING_DATE"],
    value_vars=["pred1", "pred2", "pred3", "pred4", "pred5", "pred6"],
    var_name="pred",
    value_name="value",
)

for i in tqdm(range(len(df_out)), desc="Renaming pred period as date"):
    posting_date = df_out.iloc[i]["POSTING_DATE"]  # convert to datetime object
    pred_period = df_out.iloc[i]["pred"]

    if pred_period.startswith("pred"):
        pred_num = int(
            pred_period[4:]
        )  # extract the number from the pred period string
        pred_date = posting_date + pd.DateOffset(
            months=pred_num
        )  # add the offset to the posting date
        df_out.at[i, "PRED_DATE"] = pred_date.strftime("%Y-%m")

df_out.rename(
    columns={
        "POSTING_DATE": "PRED_PERIOD",
        "PRED_DATE": "POSTING_DATE",
        "CCH": "value",
    },
    inplace=True,
)


df_out["POSTING_DATE"] = pd.to_datetime(df_out["POSTING_DATE"])

df2 = pd.merge(
    df[["CUSTOMER_SHIPTO", "POSTING_DATE", "CCH"]],
    df_out[["CUSTOMER_SHIPTO", "POSTING_DATE", "value"]],
    on=["CUSTOMER_SHIPTO", "POSTING_DATE"],
    how="outer",
)
df2["predicted"] = df2["value"].fillna(df2["CCH"])

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from scipy.signal import correlate


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

    mean_before = np.trim_zeros(series[-24:-12], "f").mean()
    mean_after = np.trim_zeros(series[-12:], "f").mean()

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
        mean_before,
        mean_after,
    )


fourier_results = df2.groupby("CUSTOMER_SHIPTO")["predicted"].apply(find_transform)
fourier_df = pd.DataFrame(
    fourier_results.tolist(),
    index=fourier_results.index,
    columns=[
        "diff_phase",
        "amplitude",
        "period_after",
        "period_before",
        "seasonal_after",
        "seasonal_before",
        "seasonal_shift",
        "mean_before",
        "mean_after",
    ],
)
df3 = pd.merge(
    df_p,
    fourier_df.reset_index(),
    how="left",
)
mask = np.abs(df3["amplitude"]) > 0.02
df3.loc[mask, "miss_period"] = (
    df3.loc[mask, "period_before"] - df3.loc[mask, "period_after"] - 1
)
df3["miss_period"].fillna(0, inplace=True)

df3["magnitute_drop"] = (df3["mean_after"] - df3["mean_before"]) / df3["mean_before"]
df3["magnitute_drop"].fillna(0, inplace=True)
df3 = df3.reset_index(drop=True)
df3 = df3.replace([-np.inf, np.inf], -100)

# get the yyyymmdd
from datetime import datetime
now = datetime.datetime.strptime(datetime.today(), "%Y-%m")


def apply_classifier(row):
    if row["percent_decline"] >= 0.25:
        slope_score = -10
        slope_t = "increase"
    elif row["percent_decline"] >= 0.2:
        slope_score = -8
        slope_t = "increase"
    elif row["percent_decline"] >= 0.15:
        slope_score = -6
        slope_t = "increase"
    elif row["percent_decline"] >= 0.1:
        slope_score = -4
        slope_t = "increase"
    elif row["percent_decline"] >= 0.05:
        slope_score = -2
        slope_t = "stay the same"
    elif row["percent_decline"] >= 0:
        slope_score = 0
        slope_t = "stay the same"
    elif row["percent_decline"] >= -0.05:
        slope_score = 2
        slope_t = "stay the same"
    elif row["percent_decline"] >= -0.1:
        slope_score = 4
        slope_t = "decrease"
    elif row["percent_decline"] >= -0.15:
        slope_score = 6
        slope_t = "decrease"
    elif row["percent_decline"] >= -0.2:
        slope_score = 8
        slope_t = "decrease"
    else:
        slope_score = 10
        slope_t = "decrease"

    if row["magnitute_drop"] >= 0.25:
        magnitute_score = -5
    elif row["magnitute_drop"] >= 0.2:
        magnitute_score = -4
    elif row["magnitute_drop"] >= 0.15:
        magnitute_score = -3
    elif row["magnitute_drop"] >= 0.1:
        magnitute_score = -2
    elif row["magnitute_drop"] >= 0.05:
        magnitute_score = -1
    elif row["magnitute_drop"] >= 0:
        magnitute_score = 0
    elif row["magnitute_drop"] >= -0.05:
        magnitute_score = 1
    elif row["magnitute_drop"] >= -0.1:
        magnitute_score = 2
    elif row["magnitute_drop"] >= -0.15:
        magnitute_score = 3
    elif row["magnitute_drop"] >= -0.2:
        magnitute_score = 4
    else:
        magnitute_score = 5

    if row["miss_period"] >= 3:
        period = -3
        period_t = "an increase"
    elif row["miss_period"] >= 2:
        period = -2
        period_t = "an increase"
    elif row["miss_period"] >= 1:
        period = 1
        period_t = "no change"
    elif row["magnitute_drop"] >= 0:
        period = 0
        period_t = "no change"
    elif row["magnitute_drop"] >= -1:
        period = 1
        period_t = "no change"
    elif row["magnitute_drop"] >= -2:
        period = 2
        period_t = "a missed"
    else:
        period = 3
        period_t = "a missed"

    if row["max_cch"] > 50:
        cch = 10
    elif row["max_cch"] > 10:
        cch = 8
    else:
        cch = 6
    risk_score = cch * (slope_score + magnitute_score + period) / 1.8

    if row["cross_zero"] > 0:
        crossing = f'Likely cross 0 by {int(row["cross_zero"])} months.'
    else:
        crossing = ""

    risk_value = f"""6 mth CCH likely {slope_t} within {int(np.abs(row["percent_decline"])*100)}%. {crossing} CCH values: {int(row["pred1"])}, {int(row["pred2"])}, {int(row["pred3"])}, {int(row["pred4"])}, {int(row["pred5"])}, {int(row["pred6"])}. {period_t} in periodicity of {period} mth."""

    risk_score = (risk_score + 100) / 2

    return round(risk_score, 0), risk_value


# apply the function to each row
df3[["risk_score", "risk_value"]] = df3.apply(
    apply_classifier, axis=1, result_type="expand"
)
len(df3["risk_value"][13])

df3.to_csv("temp_2.csv")
