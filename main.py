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
import polars as pl
import re
from dateutil.parser import parse

# from scipy.signal import correlate


new_date = datetime.today() - relativedelta(months=1)
new_date = new_date.replace(day=1)
report_date = new_date.strftime("%Y%m")
new_date = new_date.strftime("%Y-%m") + "-01"
current_date = datetime.today().replace(day=1)
# current_date = current_date - relativedelta(days=1)
min_date = current_date - relativedelta(months=24)
current_month = current_date
current_date = current_date.strftime("%Y-%m") + "-01"
min_date = min_date.strftime("%Y-%m") + "-01"
# current_date = "2023-11-01"
# For 1st stage - ts prediction
# with open("predictor12.pkl", "rb") as f:
#     model_ts = dill.load(f)

with open("model3_ts.pkl", "rb") as f:
    model_ts = dill.load(f)

# For 2nd stage - ts across period #model_lFor 2
# with open("model_list6.pkl", "rb") as f:
#     model_class = dill.load(f)
with open("model_list6_2024.pkl", "rb") as f:
    model_class = dill.load(f)

# with open("gbm_explainer2.pkl", "rb") as f:
#     gbm_explainer = dill.load(f)
with open("gbm_explainer_2024.pkl", "rb") as f:
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
# df = pd.read_pickle('df_new.pkl')
object_columns = ['CUSTOMER_NUMBER', 'SHIP_TO', 'INDUSTRY', 'INDUSTRY_SUBLEVEL', 'PLANT', 'CUSTOMER_SHIPTO']
for col in object_columns:
    if col in df.columns:
        df[col] = df[col].astype(str)
df = helper.clean_data(df)
print('Finished cleaning data')
df = helper.create_lags(df)
# create a column checking if there is a 12 preceeding period
df['less12'] = df.groupby('CUSTOMER_SHIPTO')['POSTING_DATE'].transform(lambda x: x.shift(12))
df['less12'] = df['less12'].notnull().astype(int)
print('Finished creating lag')

df.to_pickle('df.pkl')

# df = pd.read_pickle('df.pkl')

df_time = modeling_ts.predict_timeseries(df, model_ts, start_date=new_date)
df_time.to_pickle('df_time.pkl')
print('Finished modeling_ts')
# df_time = pd.read_pickle('df_time.pkl')

# create a list of MS date in 'yyyy-mm-01'
# min_date = df['POSTING_DATE'].min()
# max_date = df['POSTING_DATE'].max()
# date_object = parse(f'{max_date}-01')
# max_date = date_object - relativedelta(months=4)
# date_object = parse(f'{min_date}-01')
# min_date = date_object + relativedelta(months=6)

# list_date = pd.date_range(start=min_date.strftime('%Y-%m-%d'), end=max_date.strftime('%Y-%m-%d'), freq="MS").strftime("%Y-%m-%d")

# # randomly select 30% of distinct Customer
# from sklearn.model_selection import train_test_split
# sample_customer = df["CUSTOMER_SHIPTO"].unique()
# sample_customer = np.random.choice(sample_customer, int(len(sample_customer) * 0.65), replace=False)
# df2 = df[df["CUSTOMER_SHIPTO"].isin(sample_customer)]
# df2.reset_index(drop=True, inplace=True)
# # loop and append df
# df_time_all = pd.DataFrame()
# for i in tqdm(list_date):
#     df_time_2 = modeling_ts.predict_timeseries(df2, model_ts, start_date=i)
#     df_time_all = df_time_all.append(df_time_2)
# df_time_all.to_pickle('df_time_all.pkl')

# df_time_all["POSTING_DATE"] = df_time_all["PREDICTION_DATE"]
# df_time_all["POSTING_DATE"] = pd.to_datetime(df_time_all["POSTING_DATE"])
# df["POSTING_DATE"] = pd.to_datetime(df["POSTING_DATE"])
# df_all = df_time_all.merge(df, on=["CUSTOMER_SHIPTO","POSTING_DATE"], how='inner')
# df_all.to_pickle('df_all.pkl')

df_p = modeling_class.predict_classifer(
    df, df_time, model_class,
    start_date=new_date
)

df = pl.from_pandas(df)
def calculate_period_diff(s: pl.Series) -> pl.Series:
    return s.diff().fill_null(0)

# Group by 'CUSTOMER_SHIPTO', calculate the difference in 'period'
df = df.with_columns(
    calculate_period_diff(pl.col("period")).over("CUSTOMER_SHIPTO").alias("period_diff")
)

# Create a conditional column based on the value of 'period_diff'
df = df.with_columns(
    pl.when(pl.col("period_diff") <= -3).then(-3)
    .when(pl.col("period_diff") <= -2).then(-2)
    .when(pl.col("period_diff") <= -1).then(-1)
    .when(pl.col("period_diff") <= 0.5).then(0)
    .when(pl.col("period_diff") <= 1.5).then(1)
    .when(pl.col("period_diff") <= 2.5).then(2)
    .when(pl.col("period_diff") > 2.5).then(3)
    .otherwise(0)  # Default case if none of the above conditions are met
    .alias("miss_period")
)
df = df.to_pandas()
df['miss_period'].hist()

print('Finished modeling class')
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

def find_transform(series: pl.Series) -> tuple:
    series = series.drop_nulls()
    series_len = len(series)

    # Initialize default values
    best_period = best_period2 = diff_phase = amplitude_max = seasonal_shift = None
    result = result2 = []
    mean_before = mean_after = None

    # Calculate ACF values only if series length is sufficient
    if series_len >= 18:
        acf_values = acf(series[-12:], nlags=6, fft=True)
        acf_values2 = acf(series[-18:-6], nlags=6, fft=True)
        best_period = np.argmax(acf_values[1:]) + 1
        best_period2 = np.argmax(acf_values2[1:]) + 1
    else:
        acf_values = acf_values2 = np.array([])

    # Calculate means based on series length
    if series_len <= 24:
        middle_index = series_len // 2
        mean_before = series[:middle_index].mean()
        mean_after = series[middle_index:].mean()
    else:
        mean_before = series[-24:-12].mean()
        mean_after = series[-12:].mean()

    # Convert to pandas Series to use statsmodels (no direct Polars equivalent)
    series_pd = series.to_pandas()

    # Perform seasonal decomposition if ACF values are available
    if np.any(acf_values) and np.any(acf_values2):
        result = seasonal_decompose(series_pd[-12:], model="additive", period=best_period).seasonal
        result2 = seasonal_decompose(series_pd[-18:-6], model="additive", period=best_period2).seasonal

        diff_phase = best_period - best_period2
        amplitude_max = result2.max()
        seasonal_shift = np.argmax(np.correlate(result, result2, mode='valid'))

        result = result.tolist()
        result2 = result2.tolist()

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

# Assuming df2 is a Polars DataFrame with the necessary structure
df2 = pl.from_pandas(df2)
# Apply the function to each group
fourier_results = df2.groupby("CUSTOMER_SHIPTO").agg(pl.col("predicted").apply(find_transform))
fourier_results = fourier_results.to_pandas()

fourier_df = pd.DataFrame(
    fourier_results,
    index=fourier_results.index,
    columns=[
        'CUSTOMER_SHIPTO',
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
        slope_score = -12
        slope_t = "increase"
    elif row["percent_decline"] >= 0.2:
        slope_score = -12
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
        slope_score = 8
        slope_t = "decrease"
    elif row["percent_decline"] >= -0.2:
        slope_score = 10
        slope_t = "decrease"
    elif row["percent_decline"] >= -0.25:
        slope_score = 12
        slope_t = "decrease"
    else:
        slope_score = 12
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
        magnitute_score = 0.5
    elif row["magnitute_drop"] >= -0.1:
        magnitute_score = 1
    elif row["magnitute_drop"] >= -0.15:
        magnitute_score = 1.5
    elif row["magnitute_drop"] >= -0.2:
        magnitute_score = 2
    elif row["magnitute_drop"] >= -0.25:
        magnitute_score = 2.5
    elif row["magnitute_drop"] >= -0.3:
        magnitute_score = 3
    elif row["magnitute_drop"] >= -0.35:
        magnitute_score = 3.5
    elif row["magnitute_drop"] >= -0.4:
        magnitute_score = 4
    elif row["magnitute_drop"] >= -0.45:
        magnitute_score = 4.5
    elif row["magnitute_drop"] >= -0.5:
        magnitute_score = 5
    else:
        magnitute_score = 5

    if row["miss_period"] >= 1.5:
        period = -3
        period_t = "Increase"
    elif row["miss_period"] >= 1:
        period = -2
        period_t = "Increase"
    elif row["miss_period"] >= 0.5:
        period = 1
        period_t = "Increase"
    elif row["miss_period"] >= 0:
        period = 0
        period_t = "No change"
    elif row["miss_period"] >= -0.5:
        period = 1
        period_t = "Missed"
    elif row["miss_period"] >= -1.5:
        period = 2
        period_t = "Missed"
    else:
        period = 3
        period_t = "Missed"

    if row["max_cch"] > 100:
        cch = 8
    elif row["max_cch"] > 50:
        cch = 6
    elif row["max_cch"] > 20:
        cch = 5
    elif row["max_cch"] > 10:
        cch = 4
    else:
        cch = 3

    risk_score = cch * (1*slope_score + 1.6*magnitute_score + period) / (1*1.2+1.6*0.5+0.15)
    if risk_score is None:
        risk_score = 0
    if row["cross_zero"] > 0:
        crossing = f' Churn likely in {int(row["cross_zero"])} mth. '
    else:
        crossing = ""
    period_text =""
    if period != 0:
        period_text = f"{period_t} pred purch of {abs(period)} mth."

    risk_score = (risk_score + 100) / 2

    risk_value = f"""{report_date}: Current CCH:{int(row["CCH"])}, Last mth CCH:{int(row["CCH_lag_1"])}, 6 mth CCH likely {slope_t} within {int(np.abs(row["percent_decline"])*100)}%.{crossing} {period_text}"""

    return round(risk_score, 0), risk_value


# apply the function to each row
df3 = df3.fillna(0)
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

print("opportunities:",len(df4),"risk:",len(df5))



# df5[df5['CUSTOMER_SHIPTO']=='PEA300.AU10.0001315838']
# df_time2 = df_time.reset_index()

df5 = re.sub('{end_date}', current_date, sqlFile)

# Assuming df is your DataFrame and 'column_name' is the column where you want to make replacements
# df5['risk_value'] = df5['risk_value'].str.replace(r'20240330:', r'202402:', regex=True)

have_action = pd.read_csv('saved_risk.csv')
no_action = pd.read_csv('saved_risk_all.csv')

#create a column to check if there is action
df['action'] = None
df.loc[df['CUSTOMER_SHIPTO'].isin(have_action['SAPCustomerNumber__c']), 'action'] = 1
df.loc[df['CUSTOMER_SHIPTO'].isin(no_action['SAPCustomerNumber__c']), 'action'] = 0

#filter out None
df = df[df['action'].notnull()]
df.to_csv('temp_evaluation_old.csv')