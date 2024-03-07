"""
This script performs predictive purchase analysis using time series modeling and classification.
It retrieves data from a SQL database, applies time series prediction models, and performs classification based on various features.
The script also includes functions for data cleaning, feature engineering, and scoring.

The main steps of the script are as follows:
1. Import necessary libraries and modules.
2. Set the current date and minimum date for data retrieval.
3. Load the time series prediction model and classification model from pickle files.
4. Define a function to retrieve data from a SQL database.
5. Read an SQL file and replace placeholders with the current and minimum dates.
6. Retrieve data from the SQL database using the modified SQL query.
7. Clean the retrieved data and create lag features.
8. Save the cleaned data to a pickle file.
9. Apply time series prediction to the cleaned data using the time series model.
10. Save the predicted time series data to a pickle file.
11. Apply classification to the cleaned data and predicted time series data using the classification model.
12. Create a scoring dataframe based on the classification results.
13. Perform Fourier analysis on the predicted values to extract seasonal patterns and other features.
14. Merge the scoring dataframe with the Fourier analysis results.
15. Apply scoring rules to calculate scores for the classification features.
16. Save the final scoring dataframe.
"""

from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import modeling_class
from sqlalchemy import create_engine, text
import helper
import modeling_ts
import dill
import numpy as np
import pandas as pd
from scipy import stats
import pyodbc
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
import re
# from scipy.signal import correlate


new_date = datetime.today() - relativedelta(months=1)
new_date = new_date.replace(day=1)
report_date = new_date.strftime("%Y%m")
new_date = new_date.strftime("%Y-%m") + "-01"
current_date = datetime.today().replace(day=1)
# current_date = current_date - relativedelta(days=1)
min_date = current_date - relativedelta(months=24)
report_date = current_date.strftime("%Y%m") + "30"
current_month = current_date
current_date = current_date.strftime("%Y-%m") + "-01"
min_date = min_date.strftime("%Y-%m") + "-01"
# current_date = "2023-11-01"
# For 1st stage - ts prediction
with open("predictor12.pkl", "rb") as f:
    model_ts = dill.load(f)

# For 2nd stage - ts across period #model_lFor 2
with open("model_list6.pkl", "rb") as f:
    model_class = dill.load(f)

with open("gbm_explainer2.pkl", "rb") as f:
    gbm_explainer = dill.load(f)

def get_data(sql):
    # Old method -- using pyodbc -- deprecated
    conn = pyodbc.connect(
        """DRIVER={ODBC Driver 17 for SQL Server};
        SERVER=mlggae00sql005.linde.lds.grp,1433;
        DATABASE=APAC_DATA_REPO;
        UID=Customers_at_Risk_PRD;
        PWD=sTzmxZ3oLKRt64QoJ;
        Encrypt=no;
        TrustServerCertificate=no;
        Connection Timeout=300;
        ColumnEncryption=Enabled;
        DisableCertificateVerification=yes;"""
    )
    df = pd.read_sql(sql, conn)
    conn.close()
    return df

fd = open(r'sql\raw_data2.sql', 'r')
sqlFile = fd.read()
fd.close()

sqlFile = re.sub('{start_date}', min_date, sqlFile)
sqlFile = re.sub('{end_date}', current_date, sqlFile)

# date_max = get_data(f"SELECT MAX(CAST([BUDAT] AS DATETIME)) TIME FROM [APAC_DATA_REPO].[dbo].[PEA_CE10COC] WHERE CAST([BUDAT] AS DATETIME) <= '{current_date}' AND CAST([BUDAT] AS DATETIME) >= '{min_date}' AND [LAND1] ='AU'")
# print(date_max)
df = get_data(sqlFile)
df.to_pickle('df_new.pkl')

# df.columns

df = helper.clean_data(df)
print('Finished cleaning data')
df = helper.create_lags(df)
print('Finished creating lag')
df.to_pickle('df.pkl')
# df = pd.read_pickle('df.pkl')


df_time = modeling_ts.predict_timeseries(df, model_ts, start_date=new_date)
df_time.to_pickle('df_time.pkl')
print('Finished modeling_ts')
# df_time = pd.read_pickle('df_time.pkl')

# df_time = df_time2.copy()
df_time.reset_index(inplace=True)

df_p = modeling_class.predict_classifer(
    df, df_time, model_class,
    start_date=new_date
)
df_explainer = df_p.copy()
df_p = df_p[
        ["CUSTOMER_SHIPTO", "POSTING_DATE", "CCH", "CCH_lag_1", "CCH_lag_2",
        "slope", "percent_decline", "max_cch", "p_value", "cross_zero", "drop",
        "pred1","pred2","pred3","pred4","pred5","pred6"]
    ]

def make_scoring(df_p,df):
    df_out = pd.melt(
        df_p,
        id_vars=["CUSTOMER_SHIPTO", "POSTING_DATE"],
        value_vars=["pred1", "pred2", "pred3", "pred4", "pred5", "pred6"],
        var_name="pred",
        value_name="value",
    )

    # for i in tqdm(range(len(df_out)), desc="Renaming pred period as date"):
    #     posting_date = df_out.iloc[i]["POSTING_DATE"]  # convert to datetime object
    #     pred_period = df_out.iloc[i]["pred"]

    #     if pred_period.startswith("pred"):
    #         pred_num = int(
    #             pred_period[4:]
    #         )  # extract the number from the pred period string
    #         pred_date = posting_date + pd.DateOffset(
    #             months=pred_num
    #         )  # add the offset to the posting date
    #         df_out.at[i, "PRED_DATE"] = pred_date.strftime("%Y-%m")
    # Convert 'pred' column to integer type after stripping 'pred' from the start
    df_out['pred_num'] = df_out['pred'].str.slice(start=4).astype(int)
    # Convert 'POSTING_DATE' to datetime if it's not already
    df_out['POSTING_DATE'] = pd.to_datetime(df_out['POSTING_DATE'])
    # Use vectorized operation to add offset to 'POSTING_DATE'
    # df_out['PRED_DATE'] = (df_out['POSTING_DATE'] + pd.offsets.MonthBegin(df_out['pred_num'])).dt.strftime('%Y-%m')
    df_out['PRED_DATE'] = df_out.apply(lambda row: (row['POSTING_DATE'] + pd.offsets.MonthBegin(int(row['pred_num']))).strftime('%Y-%m'), axis=1)
    # Filter rows where 'pred' starts with 'pred'
    df_out = df_out[df_out['pred'].str.startswith('pred')]
    ###############

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
        slope_t = "decrease"
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
    risk_score = cch * (slope_score + magnitute_score + period) / (1.0+1.0+0.3)

    if row["cross_zero"] > 0:
        crossing = f' Churn likely in {int(row["cross_zero"])} mth. '
    else:
        crossing = ""

    risk_score = (risk_score + 100) / 2

    risk_value = f"""{report_date}: Current CCH:{int(row["CCH"])}, Last mth CCH:{int(row["CCH_lag_1"])}, 6 mth CCH likely {slope_t} within {int(np.abs(row["percent_decline"])*100)}%.{crossing} {period_t} pred purch of {abs(period)} mth."""

    return round(risk_score, 0), risk_value


# apply the function to each row
df3[["risk_score", "risk_value"]] = df3.apply(
    apply_classifier, axis=1, result_type="expand"
)


df3 = df3[['CUSTOMER_SHIPTO',"risk_score", "risk_value"]]
df3["risk_score"].hist()
df4 = df3[df3["risk_score"]>65]

df_explainer = df_explainer[df_explainer['CUSTOMER_SHIPTO'].isin(df4.CUSTOMER_SHIPTO)]
df5 = modeling_class.explainer(df_explainer, gbm_explainer)

df5 = df5.merge(df4, how='left')

def apply_words(row):
    return f"""{row["risk_value"]} {row["explainer"]}"""

df5["risk_value"] = df5.apply(
    apply_words, axis=1, result_type="expand"
)

df5 = df5[['CUSTOMER_SHIPTO',"risk_score", "risk_value"]]


df5.to_csv("temp_5.csv")
df4 = df3[df3["risk_score"]<45]
df4.to_csv("temp_6.csv")

len(df4)
len(df5)

# df_time2[df_time2['CUSTOMER_SHIPTO']=='PEA300.AU10.0001315838']
# df_time2 = df_time.reset_index()

df5 = re.sub('{end_date}', current_date, sqlFile)

# Assuming df is your DataFrame and 'column_name' is the column where you want to make replacements
df5['risk_value'] = df5['risk_value'].str.replace(r'2024-02:', r'202402:', regex=True)