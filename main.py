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
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
# from scipy.signal import correlate

current_date = datetime.today() - relativedelta(days=40)
min_date = current_date - relativedelta(months=24)
report_date = current_date.strftime("%Y%m") + "30"
current_month = current_date.strftime("%Y-%m")
current_date = current_date.strftime("%Y-%m") + "-01"
min_date = min_date.strftime("%Y-%m") + "-01"
current_date = "2023-11-01"
# For 1st stage - ts prediction
with open("predictor12.pkl", "rb") as f:
    model_ts = dill.load(f)

# For 2nd stage - ts across period
with open("model_list3c.pkl", "rb") as f:
    model_class = dill.load(f)

with open("gbm_explainer.pkl", "rb") as f:
    gbm_explainer = dill.load(f)

dtypes = {"CUSTOMER_NUMBER": str}
df = pd.read_csv("data/data_8.csv")
df["POSTING_DATE"] = pd.to_datetime(df["POSTING_DATE"])
# df.head()
len(df)
print(df["POSTING_DATE"].max())
df = df[df["POSTING_DATE"]<='2023-11-01']
df = df[df["POSTING_DATE"]>='2022-11-01']

# df.columns

df = helper.clean_data(df)
df = helper.create_lags(df)
df.to_pickle('df.pkl')
df = pd.read_pickle('df.pkl')


df_time = modeling_ts.predict_timeseries(df, model_ts, start_date=current_date)
df_time.to_pickle('df_time.pkl')
df_time = pd.read_pickle('df_time.pkl')

# df_time = df_time2.copy()
df_time.reset_index(inplace=True)
df_time["CUSTOMER_SHIPTO"][0]

df_time[df_time["CUSTOMER_SHIPTO"] == "PEA300.AU10.0001628667"]

# df_time = pd.read_csv("temp.csv")
df_p = modeling_class.predict_classifer(
    df, df_time, model_class,gbm_explainer,
    start_date=current_date,
    percent_1=0.05,
    percent_2=0.03,
    max_cch=5,
    PRODUCT_SALES=1500
)



def make_scoring(df_p,df):
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

    return df2

df2 = make_scoring(df_p,df)

def find_transform(series):
    series = series.dropna()
    series_len = len(series)

    if series_len >= 18:
        acf_values = acf(series[-12:], nlags=6)
        acf_values2 = acf(series[-18:-6], nlags=6)
    else:
        acf_values = []
        acf_values2 = []

    if series_len <= 24:
        middle_index = len(series) // 2
        mean_before = series[:middle_index].mean()
        mean_after = series[middle_index:].mean()
    else:
        mean_before = series[-24:-12].mean()
        mean_after = series[-12:].mean()

    if len(acf_values) > 0 and len(acf_values2) > 0:
        best_period = np.argmax(acf_values[1:]) + 1
        best_period2 = np.argmax(acf_values2[1:]) + 1

        result = seasonal_decompose(series[-12:], model="additive", period=best_period)
        result2 = seasonal_decompose(series[-18:-6], model="additive", period=best_period2)

        diff_phase = best_period - best_period2
        amplitude_max = result2.seasonal.max()
        seasonal_shift = np.argmax(
            np.correlate(result.seasonal.tolist(), result2.seasonal.tolist(), mode='valid')
        )
        result = result.seasonal.tolist()
        result2 = result2.seasonal.tolist()
    else:
        best_period = best_period2 = diff_phase = amplitude_max = seasonal_shift = None
        result = result2 = []

    return (
        diff_phase,
        amplitude_max,
        best_period,
        best_period2,
        result,
        result2,
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
mask = np.abs(df3["amplitude"]) > 0.025
df3.loc[mask, "miss_period"] = (
    df3.loc[mask, "period_before"] - df3.loc[mask, "period_after"] - 1
)
df3["miss_period"].fillna(0, inplace=True)

df3["magnitute_drop"] = (df3["mean_after"] - df3["mean_before"]) / df3["mean_before"]
df3["magnitute_drop"].fillna(0, inplace=True)
df3 = df3.reset_index(drop=True)
df3 = df3.replace([-np.inf, np.inf], -100)

# get the yyyymmdd
def apply_classifier(row):
    if row["percent_decline"] >= 0.25:
        slope_score = -10
        slope_t = "increase"
    elif row["percent_decline"] >= 0.2:
        slope_score = -10
        slope_t = "increase"
    elif row["percent_decline"] >= 0.15:
        slope_score = -10
        slope_t = "increase"
    elif row["percent_decline"] >= 0.1:
        slope_score = -8
        slope_t = "increase"
    elif row["percent_decline"] >= 0.05:
        slope_score = -4
        slope_t = "increase"
    elif row["percent_decline"] >= 0:
        slope_score = -2
        slope_t = "increase"
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
    elif row["percent_decline"] >= -0.25:
        slope_score = 10
        slope_t = "decrease"
    else:
        slope_score = 10
        slope_t = "decrease"

    if row["magnitute_drop"] >= 0.5:
        magnitute_score = -10
    elif row["magnitute_drop"] >= 0.45:
        magnitute_score = -9
    elif row["magnitute_drop"] >= 0.4:
        magnitute_score = -8
    elif row["magnitute_drop"] >= 0.35:
        magnitute_score = -7
    elif row["magnitute_drop"] >= 0.3:
        magnitute_score = -6
    elif row["magnitute_drop"] >= 0.25:
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
    elif row["magnitute_drop"] >= -0.25:
        magnitute_score = 5
    elif row["magnitute_drop"] >= -0.3:
        magnitute_score = 6
    elif row["magnitute_drop"] >= -0.35:
        magnitute_score = 7
    elif row["magnitute_drop"] >= -0.4:
        magnitute_score = 8
    elif row["magnitute_drop"] >= -0.45:
        magnitute_score = 9
    elif row["magnitute_drop"] >= -0.5:
        magnitute_score = 10
    else:
        magnitute_score = 10

    if row["miss_period"] >= 3:
        period = -3
        period_t = "Increase"
    elif row["miss_period"] >= 2:
        period = -2
        period_t = "Increase"
    elif row["miss_period"] >= 1:
        period = 1
        period_t = "Increase"
    elif row["miss_period"] >= 0:
        period = 0
        period_t = "No change"
    elif row["miss_period"] >= -1:
        period = 1
        period_t = "Missed"
    elif row["miss_period"] >= -2:
        period = 2
        period_t = "Missed"
    else:
        period = 3
        period_t = "Missed"

    if row["max_cch"] > 100:
        cch = 10
    elif row["max_cch"] > 50:
        cch = 8
    elif row["max_cch"] > 20:
        cch = 6
    elif row["max_cch"] > 10:
        cch = 4
    else:
        cch = 2
    risk_score = cch * (slope_score + magnitute_score + period) / (1.0+1.5+0.3)

    if row["cross_zero"] > 0:
        crossing = f' Churn likely in {int(row["cross_zero"])} mth. '
    else:
        crossing = ""

    risk_value = f"""{report_date}: Current CCH:{int(row["CCH"])}, Last mth CCH:{int(row["CCH_lag_1"])}, 6 mth CCH likely {slope_t} within {int(np.abs(row["percent_decline"])*100)}%.{crossing}{row["explainer"]} {period_t} pred purch of {abs(period)} mth."""

    risk_score = (risk_score + 100) / 2

    return round(risk_score, 0), risk_value


# apply the function to each row
df3[["risk_score", "risk_value"]] = df3.apply(
    apply_classifier, axis=1, result_type="expand"
)



df3 = df3[['CUSTOMER_SHIPTO',"risk_score", "risk_value"]]
len(df3)

df3.to_csv("temp_4.csv")
